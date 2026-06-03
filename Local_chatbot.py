import streamlit as st
import docx2txt
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def get_models():
    embed_model = SentenceTransformer('jhgan/ko-sroberta-multitask', device="cpu")
    llm = Ollama(model="qwen2.5:0.5b", num_predict=512, temperature=0, num_ctx=1024)
    return embed_model, llm

embed_model, llm = get_models()

st.title("사규 챗봇 ")

if "db" not in st.session_state: st.session_state.db = None

with st.sidebar:
    st.header("사규 파일 관리")
    uploaded = st.file_uploader("Word 파일 업로드", type=["docx"])
    if uploaded and st.button("로컬 인덱싱 시작"):
        text = docx2txt.process(uploaded)
        # [개선] 텍스트를 더 작은 단위로 쪼개고 겹치게 하여 맥락 유지
        lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 10]
        # 유사도를 높이기 위해 2줄씩 묶어서 처리
        paragraphs = [" ".join(lines[i:i+2]) for i in range(len(lines)-1)]
        embeddings = embed_model.encode(paragraphs)
        st.session_state.db = {"text": paragraphs, "vec": embeddings}
        st.success(f"{len(paragraphs)}개 항목 인덱싱 완료")

if prompt := st.chat_input("사규에 대해 질문하세요..."):
    if st.session_state.db:
        q_vec = embed_model.encode([prompt])
        scores = cosine_similarity(q_vec, st.session_state.db["vec"])[0]
        
        # [개선] 유사도 상위 2개를 가져와서 더 풍부한 정보 제공
        top_indices = np.argsort(scores)[-2:][::-1]
        context = "\n".join([st.session_state.db["text"][i] for i in top_indices])
        
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            # [핵심] 환각 방지 시스템 프롬프트
            full_prompt = f"""
            당신은 사규 전문 AI입니다. 아래 [지침]만을 사용하여 질문에 답변하세요.
            답변에 필요한 정보가 [지침]에 없다면 "제공된 사규에서 해당 내용을 찾을 수 없습니다."라고 답변하세요.
            [지침]: {context}
            질문: {prompt}
            답변:"""
            
            st.write_stream(llm.stream(full_prompt))
            
            # [디버깅] AI가 무엇을 근거로 답했는지 확인
            with st.expander("AI가 참고한 문서 확인"):
                st.write(context)
    else:
        st.warning("왼쪽 사이드바에서 파일을 먼저 업로드하세요.")