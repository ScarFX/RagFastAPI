from langchain_community.document_loaders import (
    WebBaseLoader,
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
    YoutubeLoader
)
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
async def load_file(filename:str,filepath:Path,metadata:dict, file_id:str = None):    
    file_ext = filename.split(".")[-1].lower()
    docs = [] 
    splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False
            )
    match file_ext: 
        case "txt":
            loader = TextLoader(filepath, autodetect_encoding=True)
            wholeDoc = loader.load()
            docs = splitter.split_documents(wholeDoc)

        case "csv":
            loader = CSVLoader(filepath)
            docs = loader.load()
        case "pdf":
            loader = PyPDFLoader(file_path=filepath)
            wholeDoc = loader.load()
            docs = splitter.split_documents(wholeDoc)
        case "docx":
            loader = Docx2txtLoader(filepath)
            wholeDoc = loader.load()
            docs = splitter.split_documents(wholeDoc)
        case _:
            raise TypeError(f'The file extension {file_ext} is not supported')
    for doc in docs:
        doc.metadata = doc.metadata | metadata
        if file_id:
                doc.metadata["file_id"] = file_id
    return docs

async def load_link(url:str, metadata:dict = None, file_id:str = None):
    website = url.split(".")[1].lower()
    docs = []
    splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False
            )
    match website:
        case 'youtube':
            loader = YoutubeLoader.from_youtube_url(url, add_video_info=False)
            wholeDoc = loader.load()
            docs = splitter.split_documents(wholeDoc)
        case _:  #assume html webpage
            loader = WebBaseLoader(url)
            wholeDoc = loader.load()
            docs = splitter.split_documents(wholeDoc)
    if metadata:
        for doc in docs:
            doc.metadata = doc.metadata | metadata 
    if file_id:
        for doc in docs:
            doc.metadata["file_id"] = file_id
    return docs
    