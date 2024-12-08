RAG backend using fastAPI with no gui (http://127.0.0.1:8000/docs#/default/)

Run: pip install -r requirements.txt

# Setting Up Vector Storage 
## PGVecto.rs 
1. Run the docker pgvecto-rs image, download if needed below
```bash
docker run \
  --name pgvecto-rs-demo \
  -e POSTGRES_PASSWORD=mysecretpassword \
  -p 5432:5432 \
  -d tensorchord/pgvecto-rs:pg16-v0.2.0
```
2. Setting environmental variables 
Must set `VECTOR_DB_MODEL=pgvecto_rs`

3. Env variables will be set as below for default credentials, modify if needed (no action needed for running docker image above)
`DB_HOST = localhost`,
`DB_USER = postgres`,
`DB_PASS = mysecretpassword`,
`DB_NAME = postgres`,
`DB_PORT = 5432`,

## qdrant 
Set Env variables:
  `VECTOR_DB_MODEL=Qdrant`
  `SERVER_URL="http://localhost:6333/" `

1. Start Qdrant docker image in port 6333 
2. Check out http://localhost:6333/dashboard to make sure DB working 

# Using RAG API 
## Starting Server 
1. Make sure vector storage is running (see `Setting Up Vector Storage`) 
2. Set env var`OPENAI_API_KEY` to your unique api key 
3. Run `uvicorn main:app --reload` to start fast api server
4. Rag api access at http://127.0.0.1:8000/docs#/default/ 

## RAG on text documents (.pdf, .txt, .docx)
1. Use `POST /document` endpoint
2. For `file` upload a .pdf, .txt, or .docx file 
3. Strongly recommend setting `file_id` for filter key for vector search 
4. Optional `metaJSON`  meta data example: `{"date":"1/1/2024",...}`
5. If 200 response, above document now embeded in vector storaged, able to be retrieved
6. Go to `GET /rag/` endpoint
7. Enter `query` will match vectors on meaning similarity
8. Enter `k` (defaults to 5) number of document chunks returned 
9. Optionally `filterJson` JSON, keys to filter for 
10. `file_ids` not needed but encouraged, list of file ids tied to documents. 
11. Should return text of document chunks that are most similar to query

## RAG on webpages and youtube links 
1. Use `POST /link` endpoint
2. For `link` paste webpage or youtube URL 
3. Strongly recommend setting `file_id` for filter key for vector search 
4. Optional `metaJSON`  meta data example: `{"date":"1/1/2024",...}`
5. If 200 response, above text of webpage or youtube transcript now embeded in vector storaged, able to be retrieved
6. Go to `GET /rag/` endpoint 
7. Enter `query` will match vectors on meaning similarity
8. Enter `k` (defaults to 5) number of document chunks returned 
9. Optionally `filterJson` JSON, keys to filter for 
10. `file_ids` not needed but encouraged, list of file ids tied to documents. 
11. Should return text of document chunks that are most similar to query

## Creating and Executing SQL on Excel and CSV files
1. Use `POST /table` endpoint 
2. For `file` upload an excel (.xslx) or .csv file
3. set `table_name` to meaningful and relevant table name (used for sql generation)
4. If 200 response, table is stored locally
5. Go to `GET /qa` 
6. Enter `question`
7. AI will look at table names, column names to generate and execute a SQL statement to answer given question 
