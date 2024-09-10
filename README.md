Barebones project. To implment RAG using fastAPI with no gui (http://127.0.0.1:8000/docs#/default/)

Run: pip install -r requirements.txt

Need .env with
OPENAI_API_KEY=$API_KEY
PINECONE_API_KEY=$API_KEY

1. Upload a .txt document (or multiple if you want). It will write

2. Query search from /rag/{query} a question to ask


## PGVecto.rs

```bash
docker run \
  --name pgvecto-rs-demo \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -p 5432:5432 \
  -d tensorchord/pgvecto-rs:pg16-v0.2.0
```

## qdrant 
Env variables:
  VECTOR_DB_MODEL=Qdrant
  SERVER_URL="http://localhost:6333/" 

1. Start Qdrant docker image in port 6333 
2. Check out http://localhost:6333/dashboard to make sure DB working 