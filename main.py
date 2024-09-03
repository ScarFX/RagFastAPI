from fastapi import FastAPI, UploadFile,Query,File, Depends, HTTPException
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
from pydantic import BaseModel
import asyncio



load_dotenv() 
CONCURRENT_LIMIT =int(os.getenv("CONCURRENT_LIMIT",5))
semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
#uvicorn main:app --reload
app = FastAPI()
#Global Variables
pc = PineconeMake(os.getenv("PINECONE_API_KEY"))
index_name = 'rag-fast-api' #Name Vector Space
spec = ServerlessSpec(
    cloud="aws", region="us-east-1"
)
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-4o-mini-2024-07-18'
)
# messages = [
#     SystemMessage(content="You are a helpful assistant."),
# ] not state transfer 
async def get_semaphore():
    async with semaphore:
        yield

@app.get("/")
async def root(_=Depends(get_semaphore)) -> dict[str,str]:
    return {"message": "Hello World"}

@app.get("/rag/llm/")
async def sendAI(query: str = Query(...,description="User questions for LLM"),
                 _=Depends(get_semaphore)) -> dict[str,str]:
    messages = [HumanMessage(content=query)]
    res = await chat.ainvoke(messages)
    return {"AI Response": res.content}

@app.get("/rag/")
async def getContext(query: str = Query(..., description="The query to process the uploaded documents"),
                                         _=Depends(get_semaphore)) -> dict[str,str]:
    context = similarDocs(query)
    return {"Context": context}

#ADD: upload multiple documents, more than text files 
@app.post("/documents") 
async def upload_file(file: UploadFile = File(...),
                       _=Depends(get_semaphore)) -> dict[str,str] :
    if file.content_type != 'text/plain':
        raise HTTPException(status_code=400, detail=f"{file.content_type} is an Invalid file type")
    content = await file.read() #Does read big files? 
    txt =  content.decode('utf-8')
    await embedStore(txt) #No need to await? 
    return {"filename": file.filename, "message": "File uploaded successfully"}

async def embedStore(txt: str, ) -> None: 
    existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()

]   
    if index_name not in existing_indexes:
        # if does not exist, create index

        pc.create_index(
            index_name,
            dimension=1536,  # dimensionality of ada 002
            metric='cosine',
            spec=spec
        )
        # wait for index to be initialized
        while not pc.describe_index(index_name).status['ready']:
            await asyncio.sleep(1)
    index = pc.Index(index_name)
    await asyncio.sleep(1)
    vector_store = PineconeVectorStore(index, embed_model)
    data = split_into_chunks(txt, 1024) # roughly 10ish lines of text 
    docData = [Document(
        page_content=doc,
        metadata={'source':'wikipedia'}#did not add metadata
        ) for doc in data]
    # metadata = [{'text': chunk} for chunk in data ] #integers not string chunk
    uuids = [str(uuid4()) for _ in range(len(docData))]
    await vector_store.aadd_documents(docData, ids=uuids) #asyncio.gather(...) runs at same time

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
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index, embed_model)
    results =  vector_store.similarity_search(query, k=1)
    # # get the text from the results
    # # feed into an augmented prompt
    # augmented_prompt = f"""Using the contexts below, answer the query.

    # Contexts:
    # {source_knowledge}

    # Query: {query}"""
    source_knowledge = "\n".join([res.page_content for res in results])
    return source_knowledge

