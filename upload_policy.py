# upload_policy.py
import os
import docx2txt
import pickle
import numpy as np

def parse_and_upload_word(file_path, ai_client):
    """
    Word 파일을 읽어와 순수 구글 SDK로만 임베딩 벡터를 추출한 뒤,
    LangChain 없이 순수 파이썬 내장 pickle 포맷으로 로컬에 저장합니다.
    """
    try:
        if not os.path.exists(file_path):
            return False, f"파일을 찾을 수 없습니다: {file_path}"
        
        # 1. Word 파일에서 텍스트 추출
        raw_text = docx2txt.process(file_path).strip()
        if not raw_text:
            return False, "추출된 텍스트가 없습니다."
            
        # 2. 텍스트를 줄바꿈 기준으로 단락 분할
        paragraphs = [p.strip() for p in raw_text.split('\n') if p.strip()]
        
        # 3. 구글 SDK로 순수 벡터 데이터셋 빌드 (404 완벽 차단)
        knowledge_base = []
        
        for p in paragraphs:
            response = ai_client.models.embed_content(
                model='text-embedding-004',
                contents=p
            )
            vector = response.embeddings[0].values
            
            # 단락 텍스트, 백터, 출처를 딕셔너리로 묶음
            knowledge_base.append({
                "text": p,
                "vector": np.array(vector, dtype=np.float32),
                "source": os.path.basename(file_path)
            })

        # 4. 파이썬 표준 pickle 파일로 로컬 저장
        with open("local_knowledge.pkl", "wb") as f:
            pickle.dump(knowledge_base, f)
        
        return True, f"총 {len(paragraphs)}개 단락 분석 및 로컬 데이터셋 생성 완료"
        
    except Exception as e:
        return False, str(e)