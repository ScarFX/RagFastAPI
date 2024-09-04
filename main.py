from fastapi import FastAPI, UploadFile,Query,Path, Body, File, Depends, HTTPException, Form
from dotenv import load_dotenv
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
from typing import Any,Dict, List


load_dotenv() 
#uvicorn main:app --reload
app = FastAPI()
#Global Variables
persistent_client = chromadb.PersistentClient()
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-4o-mini-2024-07-18'
)
# messages = [
#     SystemMessage(content="You are a helpful assistant."),
# ] not state transfer 


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
    for file in files:
        if file.content_type != 'text/plain':
            raise HTTPException(status_code=400, detail=f"{file.content_type} is an Invalid file type")
        content = await file.read() #Does read big files? 
        txt =  content.decode('utf-8')
        task = asyncio.create_task(embedStore(txt, metadata))
        tasks.append(task)
    await asyncio.gather(*tasks)  
    return {"filename": file.filename, "message": "File/s uploaded successfully"}

async def embedStore(txt: str, metaData ) -> None: 
    data = split_into_chunks(txt, 2048) # roughly 20ish lines of text 
    docData = [Document(
        page_content=doc,
        metadata=metaData
        ) for doc in data]
    vector_store = Chroma(
    client=persistent_client,
    collection_name="rag_vectors",
    embedding_function=embed_model
    )
    await process_documents(docData, vector_store)

async def process_documents(docData: list[Document], vector_store: Chroma,
                             batch_size: int = 20, concurrent_max: int = 10):
    semaphore = asyncio.Semaphore(concurrent_max)
    tasks = [] 
    for i in range(0, len(docData), batch_size):
        batch = docData[i: min(i+batch_size, len(docData))]
        task = asyncio.create_task(process_batch(batch, vector_store, semaphore))
        tasks.append(task)
    await asyncio.gather(*tasks)

        
async def process_batch(batch: list[Document], vector_store: Chroma, semaphore):
    uuids = [str(uuid4()) for _ in range(len(batch))]
    async with semaphore: 
        #tStart = datetime.now().time()
        await vector_store.aadd_documents(batch, ids=uuids)
    #tFinish = datetime.now().time()
    #print(f'Batch embeddings started at {tStart} and finished at {tFinish}\n')

def split_into_chunks(text: str, max_tokens: int) -> list[str]:
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 for space
        if current_length + word_length > max_tokens:
            # Finalize current chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    # Add any remaining words as a final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks

def similarDocs(query: str) -> str:
    vector_store = Chroma(
    client=persistent_client,
    collection_name="rag_vectors",
    embedding_function=embed_model
    )
    results =  vector_store.similarity_search(query, k=2)
    # # get the text from the results
    # # feed into an augmented prompt
    # augmented_prompt = f"""Using the contexts below, answer the query.

    # Contexts:
    # {source_knowledge}

    # Query: {query}"""
    source_knowledge = "\n------------------------------------------".join([res.page_content for res in results])
    return source_knowledge

