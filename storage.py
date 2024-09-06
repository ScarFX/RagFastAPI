import time
from uuid import uuid4
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from langchain_community.vectorstores.pgvecto_rs import PGVecto_rs
from pgvecto_rs.sdk.filters import meta_contains
from langchain_pinecone import PineconeVectorStore
from  pinecone import Pinecone as Pinecone
from pinecone import ServerlessSpec


class VectorStorage:
    def __init__(self, index_name: str, embed_model: OpenAIEmbeddings):
        self.embed_model = embed_model
        dbModel = os.getenv("VECTOR_DB_MODEL")
        if dbModel == 'Pinecone':
            #Implement Pinecone
            self.vectorDB = PineconeDB(index_name, embed_model)
        elif dbModel == 'Chroma':
            #Implement Chroma
            self.vectorDB = ChromaDB(index_name, embed_model)
        elif dbModel == 'pgvecto_rs':
            self.vectorDB = PGVectorDB(index_name, embed_model)
        else:
            raise Exception(f"The VECTOR_DB_MODEL environment variable: {dbModel} is not supported")
    
    async def aadd_documents(self, docBatch: list[Document]):
         uuids = [str(uuid4()) for _ in range(len(docBatch))]
         await self.vectorDB.vector_store.aadd_documents(docBatch, ids=uuids)
    
    def similarity_search(self,query:str, k:int):
        return self.vectorDB.vector_store.similarity_search(query,k)
        
    async def similarity_searchFilter(self,query:str, k:int, filter:dict[str,str], file_ids:list[str]):
         if isinstance(self.vectorDB.vector_store, PGVecto_rs):
                documents = []
                if file_ids is None or len(file_ids) == 0:
                     documents= self.vectorDB.vector_store.similarity_search(
                        query, k=k,filter=filter)
                else:
                    for file_id in file_ids:
                        filter["file_id"] = file_id
                        batch = self.vectorDB.vector_store.similarity_search(
                            query, k=k,filter=filter)
                        documents.extend(batch)
         return documents


#         return self.vectorDB.vector_store.similarity_search(
#     query, k=k, filter=meta_contains(filter)
# )

class ChromaDB:
    def __init__(self, index_name: str, embed_model: OpenAIEmbeddings):
        self.persistent_client = chromadb.PersistentClient()
        self.vector_store = Chroma(
        client=self.persistent_client,
        collection_name=index_name,
        embedding_function=embed_model
    )
class PGVectorDB:
    def __init__(self, index_name: str, embed_model: OpenAIEmbeddings):
        PORT = os.getenv("DB_PORT", 5432)
        HOST = os.getenv("DB_HOST", "localhost")
        USER = os.getenv("DB_USER", "postgres")
        PASS = os.getenv("DB_PASS", "mysecretpassword")
        DB_NAME = os.getenv("DB_NAME", "postgres")

        URL = "postgresql+psycopg://{username}:{password}@{host}:{port}/{db_name}".format(
            port=PORT,
            host=HOST,
            username=USER,
            password=PASS,
            db_name=DB_NAME,
        )

        self.vector_store= PGVecto_rs.from_collection_name(
            embedding=embed_model,
            db_url=URL,
            collection_name=index_name,
        )
        

class PineconeDB:
    def __init__(self, index_name: str, embed_model: OpenAIEmbeddings):
        self.pc = Pinecone(os.getenv("PINECONE_API_KEY"))
        self.embed_model = embed_model 
        self.spec = ServerlessSpec(cloud="aws", region="us-east-1")
        index = self.checkIfCreate(index_name)
        self.vector_store = PineconeVectorStore(index, embed_model)
    
    def checkIfCreate(self,index_name:str):
        existing_indexes = [
        index_info["name"] for index_info in self.pc.list_indexes()
    ]  
        if index_name not in existing_indexes:
            # if does not exist, create index
            self.pc.create_index(
                index_name,
                dimension=1536,  # dimensionality of ada 002
                metric='cosine',
                spec=self.spec
            )
            # wait for index to be initialized
            while not self.pc.describe_index(index_name).status['ready']:
                time.sleep(1)
        return self.pc.Index(index_name)

