from multiprocessing import process
from fastapi import (
    FastAPI,
    UploadFile,
    Query,
    Path,
    Body,
    File,
    Depends,
    HTTPException,
    Form,
)
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
from langchain.schema import HumanMessage, SystemMessage, AIMessage
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
from logger import setup_logger, LogMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from langchain.globals import set_verbose
from ETLdb import ETLdb
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.prebuilt import create_react_agent


load_dotenv()
logger = setup_logger(name="fastapi_app")
# uvicorn main:app --reload
app = FastAPI()
app.add_middleware(LogMiddleware, logger=logger)
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY","")

# Global Variables
embed_model = OpenAIEmbeddings(model="text-embedding-3-small")
index_name = "db-vectors"  # Must be lower case or '-'
vector_storage = VectorStorage(index_name, embed_model)
chat = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini-2024-07-18"
)
DB_LOCATION = '/uploadDB/etl.db' #directory must exist, but .db file not neccessary
etl = ETLdb(DB_LOCATION)


# New function to get file_id requirement from environment
@lru_cache()
def get_file_id_requirement():
    return os.getenv("STRICT_FILE_ID_REQUIREMENT", "false").lower() == "true"


# New dependency for file_id validation
def file_id_validator(
    files: List[UploadFile],
    file_ids: List[str] | None,
    strict_requirement: bool = Depends(get_file_id_requirement),
):
    if strict_requirement:
        if not file_ids or len(files) != len(file_ids):
            raise HTTPException(
                status_code=400,
                detail=f"Strict mode: Number of file_ids ({len(file_ids) if file_ids else 0}) must match number of files ({len(files)})",
            )
    return file_ids if file_ids else [None] * len(files)


@app.get("/")
async def root() -> dict[str, str]:
        logger.info("Handling root request")
        return {"message": "Hello World"}


@app.get("/rag/llm/")
async def sendAI(
    query: str = Query(..., description="User questions for LLM")
) -> dict[str, str]:
    logger.info(f"Handling sendAI request with query: {query}")
    messages = [HumanMessage(content=query)]
    res = await chat.ainvoke(messages)
    logger.info(
        f"AI response generated for query: {query} with AI Response: {res.content}"
    )
    return {"AI Response": res.content}


@app.get("/rag/")
async def getContext(
    query: str = Query(..., description="The query to process the uploaded documents"),
    k: Annotated[int, Query(description="Number of documents to return")] = 5,
    filterJson: Annotated[
        str | None, Query(description="Json of metadata to search for")
    ] = "",
    file_ids: Annotated[
        List[str] | None, Query(description="file_id to search for")
    ] = [],
) -> dict[str, str]:
    logger.info(
        f"Handling RAG request with query: {query}, k: {k}, MetaData: {filterJson}, file_ids: {file_ids}"
    )
    filter = {}
    if filterJson:
        try:
            filter.update((json.loads(filterJson)))
        except json.JSONDecodeError:
            return {"error": "Invalid JSON"}
    if not filterJson and not file_ids:  # No filter and file_id
        context = await similarDocs(query, k)
    else:
        context = await similarDocs(query, k, filter, file_ids)
    logger.info(f"The query: {query} returned context: \n\n {context}")
    return {"Context": context}

@app.get("/qa")
async def qa_database(
    question: Annotated[str, Query(..., description="The question to ask the database tables")]
) -> dict[str, str]:
    logger.info(
        f"Q/A with: question: {question}"
    )
    db = SQLDatabase.from_uri("sqlite://"+DB_LOCATION)
    toolkit = SQLDatabaseToolkit(db=db, llm=chat)
    tools = toolkit.get_tools()
    sql_prompt = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

You have access to the following tables: {table_names}
""".format(
    table_names=db.get_usable_table_names()
)
    system_message = SystemMessage(content=sql_prompt)
    agent_executor = create_react_agent(chat, tools, state_modifier=system_message)
    res = agent_executor.invoke({"messages":[HumanMessage(content=question)]})
    response = res['messages'][-1].content
    logger.info(
       f"Q/A Agent returned with:{response} to the question:\n {question}"
    )
    return {"AI Response:": response}

#Excel or CSV File, structured data 
@app.post("/table")
async def upload_table(
    file: Annotated[UploadFile, File(..., description="Excel or CSV to upload")],
    table_name: Annotated[str, Form(..., description="Meaningful name of table")]
) -> dict[str, str]: 
    logger.info(
        f"Uploading table file with filename: {file.filename}"
    )
    allowed_types = {
    "csv", "xlsx"
}
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in allowed_types:  # check type
        raise HTTPException(
            status_code=400, detail=f"{file_ext} is an Invalid file type"
        )
    file_location = Path("uploads") / file.filename  # write location
    with file_location.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    etl.fileToSQL(file_location, table_name, file_ext)

    return {"filename":file.filename, "Message":"Table successfully uploaded"}
# ADD: upload multiple documents, more than text files
@app.post("/document")
async def upload_file(
    file: Annotated[UploadFile, File(..., description="File to upload")],
    file_id: Annotated[str | None, Form(description="File id")] = "",
    vector_store_id: Annotated[str | None, Form()] = "",
    metaJson: Annotated[str | None, Form()] = "",
) -> dict[str, str]:
    logger.info(
        f"Uploading file with filename: {file.filename} with file_id: {file_id} and metadata: {metaJson}"
    )
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
        "text/csv",
        "text/plain",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    if file.content_type not in allowed_types:  # check type
        raise HTTPException(
            status_code=400, detail=f"{file.content_type} is an Invalid file type"
        )
    file_location = Path("uploads") / file.filename  # write location
    with file_location.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # load_file() is async
    docData.extend(await load_file(file.filename, file_location, metaData, file_id))
    task = asyncio.create_task(process_documents(docData))
    tasks.append(task)
    await asyncio.gather(*tasks)
    return {"filename": file.filename, "message": "File/s written successfully"}


@app.post("/documents")
async def upload_files(
    files: Annotated[List[UploadFile], File(...)],
    file_ids: Annotated[List[str], Form()] = [],
    vector_store_id: Annotated[str | None, Form()] = "",
    metaJson: Annotated[str | None, Form()] = "",
) -> dict[str, str]:

    logger.info(f"Uploading multiple files with metadata: {metaJson}")

    strict_requirement = get_file_id_requirement()

    if strict_requirement and (not file_ids or len(files) != len(file_ids)):
        raise HTTPException(
            status_code=400,
            detail=f"Strict mode: Number of file_ids ({len(file_ids) if file_ids else 0}) must match number of files ({len(files)})",
        )

    metaData = {}
    if vector_store_id:
        metaData["vector_store_id"] = vector_store_id
    if metaJson:
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
        "text/csv",
        "text/plain",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }

    for i, file in enumerate(files):
        logger.info(
            f"Uploading file in group of files ({i+1}/{len(files)}) with filename: {file.filename} "
        )
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, detail=f"{file.content_type} is an Invalid file type"
            )

        file_location = Path("uploads") / file.filename
        with file_location.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        file_id = file_ids[i] if file_ids and i < len(file_ids) else None
        cur_docs = await load_file(file.filename, file_location, metaData, file_id)
        task = asyncio.create_task(process_documents(cur_docs))
        tasks.append(task)

    await asyncio.gather(*tasks)
    return {"filename": "MULTIPLE FILES", "message": "File/s written successfully"}


# Upload a url
@app.post("/link")
async def upload_url(
    url: Annotated[str, Form(..., description="Link to pull data from")],
    file_id: Annotated[str | None, Form(description="File id")] = "",
    vector_store_id: Annotated[
        str | None, Form(description="Insert string vector id optional")
    ] = "",
    metaJson: Annotated[
        str | None, Form(description="Insert JSON metadata opptional")
    ] = "",
) -> dict[str, str]:
    logger.info(f"Uploading link: {url} with file_id: {file_id} metadata: {metaJson}")
    metaData = {}
    if vector_store_id:
        metaData["vector_store_id"] = vector_store_id
    if metaJson:
        try:
            metaData.update((json.loads(metaJson)))
        except json.JSONDecodeError:
            return {"error": "Invalid JSON"}
    docData = await load_link(url, metaData, file_id)  # Okay if file_id is None
    await process_documents(docData)
    return {"URL: ": url, "message": "Link successfully uplaoded "}


# Upload a url
@app.post("/links")
async def upload_urls(
    url_list: Annotated[List[str], Form(..., description="Links to pull data from")],
    file_ids: Annotated[
        List[str | None], Form(description="One id for each file uploaded")
    ] = [],
    vector_store_id: Annotated[
        str | None, Form(description="Insert string vector id optional")
    ] = "",
    metaJson: Annotated[
        str | None, Form(description="Insert JSON metadata opptional")
    ] = "",
) -> dict[str, str]:
    logger.info(f"Uploading list of links with metadata: {metaJson}")
    metaData = {}
    url_list = url_list[0].split(",")
    if file_ids[0] != "":
        file_ids = file_ids[0].split(",")
    if file_ids[0] != "" and (len(file_ids) != len(url_list)):
        raise HTTPException(
            status_code=400,
            detail=f"Unequal size between file_ids inputed: {len(file_ids)} and number of files: {len(url_list)}",
        )
    if vector_store_id:
        metaData["vector_store_id"] = vector_store_id
    if metaJson:
        try:
            metaData.update((json.loads(metaJson)))
        except json.JSONDecodeError:
            return {"error": "Invalid JSON"}
    tasks = []
    for i in range(len(url_list)):
        logger.info(
            f"Uploading link({i+2}/{len(url_list)}: {url_list[i]} metadata: {metaJson}"
        )
        if file_ids[0] != "":
            docData = await load_link(url_list[i], metaData, file_ids[i])
        else:
            docData = await load_link(url_list[i], metaData)
        task = asyncio.create_task(process_documents(docData))
        tasks.append(task)
    await asyncio.gather(*tasks)
    return {"URL: ": "MULTIPLE LINKS", "message": "Link successfully uplaoded "}


async def process_documents(
    docData: list[Document], batch_size: int = 20, concurrent_max: int = 10
):
    semaphore = asyncio.Semaphore(concurrent_max)
    tasks = []
    logger.info(f"Processing list of documents of length: {len(docData)}")
    for i in range(0, len(docData), batch_size):
        batch = docData[i : min(i + batch_size, len(docData))]
        logger.info(f"Sending batch {i} to {i+len(batch)} / {len(docData)}")
        task = asyncio.create_task(process_batch(batch, semaphore))
        tasks.append(task)
    await asyncio.gather(*tasks)


async def process_batch(batch: list[Document], semaphore):
    async with semaphore:
        await vector_storage.aadd_documents(batch)  # Call to wrapper method


async def similarDocs(
    query: str, k: int = 5, filter: dict[str, str] = None, file_ids: list[str] = None
) -> str:
    if filter or file_ids:
        results = await vector_storage.similarity_searchFilter(
            query, k, filter, file_ids
        )
    else:
        results = vector_storage.similarity_search(query, k)  # wrapper method
    # # get the text from the results
    # # feed into an augmented prompt
    # augmented_prompt = f"""Using the contexts below, answer the query.

    # Contexts:
    # {source_knowledge}

    # Query: {query}"""
    source_knowledge = '\n -----------------DOCUMENT-SEPERATOR------------------------------ \n'.join(
        [res.page_content for res in results]
    )
    return source_knowledge


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
