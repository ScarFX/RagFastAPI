from multiprocessing import process
from fastapi import FastAPI, UploadFile,Query,Path, Body, File, Depends, HTTPException, Form
from dotenv import load_dotenv
import uvicorn
import os
import time
from  pinecone import Pinecone as PineconeMake
from tqdm.auto import tqdm
from langchain_pinecone import PineconeVectorStore
from pinecone import ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.schema import (
    HumanMessage,SystemMessage,AIMessage)
from uuid import uuid4
import json
from pydantic import BaseModel, Field
import asyncio
from datetime import datetime
from langchain_chroma import Chroma
import chromadb
from typing import Any,Dict, List, Optional
from storage import VectorStorage
from docLoader import *
import shutil
from pathlib import Path


load_dotenv() 
#uvicorn main:app --reload
app = FastAPI()
#Global Variables
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
index_name  = "depends-db" #Must be lower case or '-' 
vector_storage = VectorStorage(index_name,embed_model)
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-4o-mini-2024-07-18'
)


@app.get("/")
async def root() -> dict[str,str]:
    return {"message": "Hello World"}

@app.get("/rag/llm/")
async def sendAI(query: str = Query(...,description="User questions for LLM")) -> dict[str,str]:
    messages = [HumanMessage(content=query)]
    res = await chat.ainvoke(messages)
    return {"AI Response": res.content}

@app.get("/rag/")
async def getContext(query: str = Query(..., description="The query to process the uploaded documents")) -> dict[str,str]:
    context = similarDocs(query)
    return {"Context": context}

#ADD: upload multiple documents, more than text files 
@app.post("/documents") 
async def upload_file(files: List[UploadFile] = (File(...)), metaJson: str= Form(...)) -> dict[str,str] :
    try:
        metadata: Dict[str, Any] = json.loads(metaJson)
    except json.JSONDecodeError:
        return {"error":"Invalid JSON"}
    tasks = []
    docData = []
    allowed_types = {
        'text/csv',
        'text/plain',
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        }
    for file in files:
        if file.content_type not in allowed_types : #check type
            raise HTTPException(status_code=400, detail=f"{file.content_type} is an Invalid file type")
        file_location = Path("uploads") / file.filename #write location
        with file_location.open('wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        #load_file() is async 
        docData.extend(await load_file(file.filename,file_location,metadata)) 
        task = asyncio.create_task(process_documents(docData))
        tasks.append(task)
    await asyncio.gather(*tasks)  
    return {"filename": file.filename, "message": "File/s written successfully"}

#Upload a url
@app.post("/links")
async def upload_url(url:str = Form(...,description="link to pull data from"), metaJson: str = Form(...)):
    try:
        metadata: Dict[str, Any] = json.loads(metaJson)
    except json.JSONDecodeError:
        return {"error":"Invalid JSON"}
    docData = await load_link(url)
    await process_documents(docData)
    return {"URL: ": url, "message": "Link successfully uplaoded "}
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


def similarDocs(query: str) -> str:
    results =  vector_storage.similarity_search(query, k=2) #wrapper method
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
