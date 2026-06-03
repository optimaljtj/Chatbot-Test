from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import docx2txt
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

app = FastAPI()
db = {"text": [], "vec": []}

embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask')
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))

class Query(BaseModel):
    prompt: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    text = docx2txt.process(file.file)
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
    db["text"] = [" ".join(lines[i:i+2]) for i in range(len(lines)-1)]
    db["vec"] = embed_model.encode(db["text"])
    return {"message": "인덱싱 완료", "count": len(db["text"])}

@app.post("/ask")
async def ask(query: Query):
    q_vec = embed_model.encode([query.prompt])
    scores = cosine_similarity(q_vec, db["vec"])[0]
    top_indices = np.argsort(scores)[-2:][::-1]
    context = "\n".join([db["text"][i] for i in top_indices])
    
    # 여기서 직접 스트리밍 응답을 반환해도 되고, 텍스트를 반환해도 됩니다.
    response = llm.invoke(f"지침: {context}\n질문: {query.prompt}")
    return {"answer": response.content, "context": context}