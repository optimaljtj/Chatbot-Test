import streamlit as st
import requests

st.title("분리형 사규 챗봇")
BACKEND_URL = "http://localhost:8000"

with st.sidebar:
    uploaded = st.file_uploader("파일 업로드", type=["docx"])
    if uploaded and st.button("업로드"):
        files = {"file": uploaded.getvalue()}
        res = requests.post(f"{BACKEND_URL}/upload", files={"file": uploaded})
        st.success(res.json()["message"])

if prompt := st.chat_input("질문하세요"):
    with st.chat_message("user"): st.markdown(prompt)
    
    # 백엔드 API 호출
    res = requests.post(f"{BACKEND_URL}/ask", json={"prompt": prompt})
    answer = res.json()["answer"]
    
    with st.chat_message("assistant"):
        st.markdown(answer)
        with st.expander("근거 문서"):
            st.write(res.json()["context"])