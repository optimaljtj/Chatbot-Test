import os
import streamlit as st
import docx2txt
import numpy as np
import warnings
import pickle 
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# .env 파일에서 환경 변수 불러오기
load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")
warnings.filterwarnings("ignore", category=DeprecationWarning)

DB_FILE = "db.pkl" # 저장할 파일명

@st.cache_resource
def get_models():
    embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask', device="cpu")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0) if google_api_key else None
    return embed_model, llm

embed_model, llm = get_models()

# 1. 앱 시작 시 파일 존재 여부 확인 후 로드
if "db" not in st.session_state:
    if os.path.exists(DB_FILE):
        with open(DB_FILE, "rb") as f:
            st.session_state.db = pickle.load(f)
    else:
        st.session_state.db = None

st.title("사규 챗봇 (데이터 파일 저장)")

with st.sidebar:
    st.header("사규 파일 관리")
    uploaded = st.file_uploader("Word 파일 업로드", type=["docx"])
    
    if uploaded and st.button("로컬 인덱싱 시작"):
        text = docx2txt.process(uploaded)
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
        paragraphs = [" ".join(lines[i:i+2]) for i in range(len(lines)-1)]
        embeddings = embed_model.encode(paragraphs)
        
        # 데이터 구조 생성
        data = {"text": paragraphs, "vec": embeddings}
        
        # 2. 파일로 저장
        with open(DB_FILE, "wb") as f:
            pickle.dump(data, f)
            
        st.session_state.db = data
        st.success("인덱싱 완료 및 파일 저장 성공!")

if prompt := st.chat_input("사규에 대해 질문하세요..."):
    if not google_api_key:
        st.error("API 키가 없어 답변을 생성할 수 없습니다.")
    elif not st.session_state.db:
        st.warning("왼쪽 사이드바에서 파일을 먼저 업로드하세요.")
    else:
        q_vec = embed_model.encode([prompt])
        scores = cosine_similarity(q_vec, st.session_state.db["vec"])[0]
        
        top_indices = np.argsort(scores)[-2:][::-1]
        context = "\n".join([st.session_state.db["text"][i] for i in top_indices])
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            full_prompt = f"""
            당신은 사규 전문 AI입니다. 아래 [지침]만을 사용하여 질문에 답변하세요.
            [지침]: {context}
            질문: {prompt}
            답변:"""
            
            # 스트리밍 답변 생성
            st.write_stream(llm.stream(full_prompt))
            
            with st.expander("AI가 참고한 문서 확인"):
                st.write(context)