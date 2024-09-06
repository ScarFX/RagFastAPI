from multiprocessing import process
from fastapi import FastAPI, UploadFile, Query, Path, Body, File, Depends, HTTPException, Form
from dotenv import load_dotenv
from traitlets import default
import uvicorn
import os
import time
from pinecone import Pinecone as PineconeMake
from tqdm.auto import tqdm
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.schema import (
    HumanMessage, SystemMessage, AIMessage)
from uuid import uuid4
import json
from pydantic import BaseModel, Field
import asyncio
from datetime import datetime
from langchain_chroma import Chroma
import chromadb
from typing import Annotated, Any, Dict, List, Optional
from storage import VectorStorage
from docLoader import *
import shutil
from pathlib import Path
from functools import lru_cache

load_dotenv()
# uvicorn main:app --reload
app = FastAPI()

# Global Variables
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
index_name = "depends-db"  # Must be lower case or '-'
vector_storage = VectorStorage(index_name, embed_model)
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-4o-mini-2024-07-18'
)

class MetaFile(BaseModel):
    file_id: str
    metaJson: dict | None = None

# New function to get file_id requirement from environment
@lru_cache()
def get_file_id_requirement():
    return os.getenv("STRICT_FILE_ID_REQUIREMENT", "false").lower() == "true"

# New dependency for file_id validation
def file_id_validator(
    files: List[UploadFile],
    file_ids: List[str] | None,
    strict_requirement: bool = Depends(get_file_id_requirement)
):
    if strict_requirement:
        if not file_ids or len(files) != len(file_ids):
            raise HTTPException(
                status_code=400,
                detail=f"Strict mode: Number of file_ids ({len(file_ids) if file_ids else 0}) must match number of files ({len(files)})"
            )
    return file_ids if file_ids else [None] * len(files)

@app.get("/")
async def root() -> dict[str,str]:
    return {"message": "Hello World"}

@app.get("/rag/llm/")
async def sendAI(query: str = Query(...,description="User questions for LLM")) -> dict[str,str]:
    messages = [HumanMessage(content=query)]
    res = await chat.ainvoke(messages)
    return {"AI Response": res.content}

@app.get("/rag/")
async def getContext(query: str = Query(..., description="The query to process the uploaded documents"),
                     k: Annotated[int, Query(description="Number of documents to return")]=5,
                     filterJson: Annotated[ str  | None, Query(description="Json of metadata to search for")] = None,
                     file_ids: Annotated[List[str]| None, Query(description="file_id to search for")] = None
                     ) -> dict[str,str]:
    filter = {} 
    if filterJson is not None :
        try:
            filter.update((json.loads(filterJson)))
        except json.JSONDecodeError:
            return {"error":"Invalid JSON"}
    if filterJson is None and file_ids is None: #No filter and file_id
        context = await similarDocs(query, k)
    else:
        context = await similarDocs(query,k,filter,file_ids )

    return {"Context": context}

#ADD: upload multiple documents, more than text files 
@app.post("/document") 
async def upload_file(
    file: Annotated[UploadFile, File(..., description="File to upload")],
    file_id: Annotated[str | None, Form(description="File id")] = "",
    vector_store_id: Annotated[str | None, Form()] = "",
    metaJson: Annotated[str| None, Form()] = "" 
) -> dict[str, str]:
    metaData = {}
    if vector_store_id:
        metaData["vector_store_id"] = vector_store_id
    if metaJson:
        try:
            metaData.update(json.loads(metaJson))
        except json.JSONDecodeError:
            return {"error": "Invalid JSON"}

    tasks = []
    docData = []
    allowed_types = {
        'text/csv',
        'text/plain',
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
    if file.content_type not in allowed_types : #check type
        raise HTTPException(status_code=400, detail=f"{file.content_type} is an Invalid file type")
    file_location = Path("uploads") / file.filename #write location
    with file_location.open('wb') as buffer:
        shutil.copyfileobj(file.file, buffer)
    #load_file() is async 
    docData.extend(await load_file(file.filename,file_location,metaData,file_id)) 
    task = asyncio.create_task(process_documents(docData))
    tasks.append(task)
    await asyncio.gather(*tasks)  
    return {"filename": file.filename, "message": "File/s written successfully"}

@app.post("/documents")
async def upload_files(
    files: Annotated[List[UploadFile], File(...)],
    file_ids: Annotated[List[str], Form()] = [],
    vector_store_id: Annotated[str | None, Form()] = "",
    metaJson: Annotated[str | None, Form()] = "" 
) -> dict[str, str]:
    strict_requirement = get_file_id_requirement()
    
    if strict_requirement and (not file_ids or len(files) != len(file_ids)):
        raise HTTPException(
            status_code=400,
            detail=f"Strict mode: Number of file_ids ({len(file_ids) if file_ids else 0}) must match number of files ({len(files)})"
        )

    metaData = {}
    if vector_store_id is not None:
        metaData["vector_store_id"] = vector_store_id
    if metaJson is not None:
        try:
            # Check if metaJson is already a dictionary
            if isinstance(metaJson, dict):
                metaData.update(metaJson)
            else:
                metaData.update(json.loads(metaJson))
        except json.JSONDecodeError:
            # If it's not valid JSON, just use it as a string
            metaData["meta"] = metaJson

    tasks = []
    allowed_types = {
        'text/csv',
        'text/plain',
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    }
    
    for i, file in enumerate(files):
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"{file.content_type} is an Invalid file type")
        
        file_location = Path("uploads") / file.filename
        with file_location.open('wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        file_id = file_ids[i] if file_ids and i < len(file_ids) else None
        cur_docs = await load_file(file.filename, file_location, metaData, file_id)
        task = asyncio.create_task(process_documents(cur_docs))
        tasks.append(task)
    
    await asyncio.gather(*tasks)  
    return {"filename": "MULTIPLE FILES", "message": "File/s written successfully"}

#Upload a url
@app.post("/link")
async def upload_url(url: Annotated[str,Form(...,description="Link to pull data from")],
                     file_id: Annotated[str | None, Form( description="File id")] = "",
                     vector_store_id: Annotated[str | None, Form(description="Insert string vector id optional")] = "", 
                     metaJson: Annotated[str | None, Form(description="Insert JSON metadata opptional")] = ""
                     ) -> dict[str,str]:
    metaData = {}
    if vector_store_id is not None:
        metaData["vector_store_id"] = vector_store_id
    if metaJson:
        try:
           metaData.update((json.loads(metaJson)))
        except json.JSONDecodeError:
            return {"error":"Invalid JSON"}
    docData = await load_link(url, metaData, file_id) #Okay if file_id is None
    await process_documents(docData)
    return {"URL: ": url, "message": "Link successfully uplaoded "}

#Upload a url
@app.post("/links")
async def upload_urls(url_list: Annotated[List[str],Form(...,description="Links to pull data from")],
                     file_ids: Annotated[List[str | None], Form(description="One id for each file uploaded")] = [],
                     vector_store_id: Annotated[str | None, Form(description="Insert string vector id optional")] = "", 
                     metaJson: Annotated[str | None, Form(description="Insert JSON metadata opptional")] = ""
                     ) -> dict[str,str]:
    metaData = {}
    url_list = url_list.split(',')
    if file_ids is not None:
        file_ids = file_ids.split(",")
    if (file_ids is not None) and (len(file_ids) != len(url_list)):
        raise HTTPException(status_code=400, detail=f"Unequal size between file_ids inputed: {len(file_ids)} and number of files: {len(url_list)}")
    if vector_store_id is not None:
        metaData["vector_store_id"] = vector_store_id
    if metaJson:
        try:
           metaData.update((json.loads(metaJson)))
        except json.JSONDecodeError:
            return {"error":"Invalid JSON"}
    tasks = []
    for i in range(url_list):
        docData = await load_link(url_list[i], metaData, file_ids[i])
        task =  asyncio.create_task(process_documents(docData))
        tasks.append(task)
    await asyncio.gather(tasks)
    return {"URL: ": "MULTIPLE LINKS", "message": "Link successfully uplaoded "}

async def process_documents(docData: list[Document],
                             batch_size: int = 20, concurrent_max: int = 10):
    semaphore = asyncio.Semaphore(concurrent_max)
    tasks = [] 
    for i in range(0, len(docData), batch_size):
        batch = docData[i: min(i+batch_size, len(docData))]
        task = asyncio.create_task(process_batch(batch, semaphore))
        tasks.append(task)
    await asyncio.gather(*tasks)

        
async def process_batch(batch: list[Document], semaphore):
    async with semaphore: 
        await vector_storage.aadd_documents(batch) #Call to wrapper method


async def similarDocs(query: str, k:int = 5,  filter: dict[str,str] = None, file_ids: list[str] = None) -> str:
    if filter is not None or file_ids is not None: 
        results = await vector_storage.similarity_searchFilter(query,k,filter,file_ids)
    else:
        results =  vector_storage.similarity_search(query,k) #wrapper method
    # # get the text from the results
    # # feed into an augmented prompt
    # augmented_prompt = f"""Using the contexts below, answer the query.

    # Contexts:
    # {source_knowledge}

    # Query: {query}"""
    source_knowledge = "\n------------------------------------------".join([res.page_content for res in results])
    return source_knowledge


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
