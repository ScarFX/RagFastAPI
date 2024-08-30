from fastapi import FastAPI, UploadFile, File
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
    HumanMessage)
from uuid import uuid4



load_dotenv() 
#uvicorn main:app --reload
app = FastAPI()
#Global Variables
pc = PineconeMake(os.getenv("PINECONE_API_KEY"))
index_name = 'rag-fast-api' #Name Vector Space
spec = ServerlessSpec(
    cloud="aws", region="us-east-1"
)
embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model='gpt-4o-mini-2024-07-18'
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/rag/{query}")
async def getContext(query):
    prompt = HumanMessage(
        content=augment_prompt(query)
    )
    messages = [prompt] #One message at a time 
    res = chat.invoke(messages)
    return {"AI Response": res.content}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read() #Does read big files? 
    txt =  content.decode('utf-8')
    embedStore(txt)
    return {"filename": file.filename, "message": "File uploaded successfully"}

def embedStore(txt): 
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
            time.sleep(1)
    index = pc.Index(index_name)
    time.sleep(1)
    vector_store = PineconeVectorStore(index, embed_model)
    data = split_into_chunks(txt, 1024) # roughly 10ish lines of text 
    docData = [Document(
        page_content=doc,
        metadata={'source':'wikipedia'}
        ) for doc in data]
    # metadata = [{'text': chunk} for chunk in data ] #integers not string chunk
    uuids = [str(uuid4()) for _ in range(len(docData))]
    vector_store.add_documents(docData, ids=uuids)

def split_into_chunks(text, max_tokens):
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

def augment_prompt(query: str):
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index, embed_model)
    results = vector_store.similarity_search(query, k=1)
    # get the text from the results
    source_knowledge = "\n".join([res.page_content for res in results])
    # feed into an augmented prompt
    augmented_prompt = f"""Using the contexts below, answer the query.

    Contexts:
    {source_knowledge}

    Query: {query}"""
    return augmented_prompt