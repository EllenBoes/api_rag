import os
import io
import sys
from dotenv import load_dotenv, find_dotenv
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from pinecone.core.client.exceptions import NotFoundException, PineconeApiException
import warnings
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import asyncio

app = FastAPI()

# Liste des origines autorisées
origins = [
    "http://localhost:4321",
    "http://127.0.0.1:4321",
]

# Ajout du middleware CORS à votre application FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

warnings.filterwarnings("ignore")

sys.path.append('../..')

_ = load_dotenv(find_dotenv(), override=True)

pc_api_key  = os.environ['PINECONE_API_KEY']

# Chargement des modèles
mistral = LlamaCpp(
    model_path= "/Users/boes/Documents/ESGI/PA-Avatar/models/mistral-7b-instruct-v0.2.Q4_0.gguf",
    temperature=0.01,
    max_tokens=2500,
    top_p=1,
)

r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=20
)

camembert = SentenceTransformerEmbeddings(model_name="/Users/boes/Documents/ESGI/PA-Avatar/models/camembert-base")

prompt = PromptTemplate.from_template("""Utilisez les éléments de contexte suivants pour répondre à la question finale.
                                      <context>
                                      {context}
                                      </context>
                                    Si vous ne connaissez pas la réponse, dites simplement que vous ne savez pas, n'essayez pas d'inventer une réponse.
                                      Formulez une réponse concise en 5 phrases maximum. 
                                      Question : {input}
                                      Réponse: """)

document_chain = create_stuff_documents_chain(mistral, prompt)

# Create class with pydantic BaseModel
class QuestionRequest(BaseModel):
    question: str

# Fonction de traitement de la requête de manière asynchrone
async def answer_question(request: str, retrieval_chain):
    result = await asyncio.to_thread(retrieval_chain.invoke, {"input": request})
    return result['answer']

@app.post("/question/")
async def rag(request: QuestionRequest, pc_index_name: str = Body(...)):
    try:
        pinecone_vs = PineconeVectorStore.from_existing_index(embedding=camembert, index_name=pc_index_name)
    except NotFoundException:
        raise HTTPException(status_code=404, detail="Index not found")
    retrieval_chain = create_retrieval_chain(pinecone_vs.as_retriever(), 
                                         document_chain)
    try:
        # Call your translation function
        answer = await answer_question(request.question, retrieval_chain)
        return {"answer": answer}
    except Exception as e:
        # Handle exceptions or errors during translation
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8080)
