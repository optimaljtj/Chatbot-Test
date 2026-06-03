import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

def test_api_connection():
    # .env 파일 로드
    load_dotenv()
    
    # 1. 환경 변수 확인
    api_key = os.getenv("GEMINI_API_KEY")
    
    print("--- 구글 API 키 로딩 테스트 ---")
    if not api_key:
        print("❌ 실패: .env 파일에서 GOOGLE_API_KEY를 찾을 수 없습니다.")
        return
    
    # 키 앞뒤 4자리만 보여주어 보안 유지
    masked_key = f"{api_key[:4]}****{api_key[-4:]}"
    print(f"✅ 성공: API 키가 로드되었습니다. (형식: {masked_key})")
    
    # 2. 모델 연결 테스트
    print("\n--- 구글 제미나이 모델 연결 테스트 ---")
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
        response = llm.invoke("안녕? 연결 테스트 중이야. 짧게 응답해줘.")
        print(f"✅ 모델 응답 성공: {response.content}")
    except Exception as e:
        print(f"❌ 실패: 모델 연결 중 오류 발생\n오류 내용: {e}")

if __name__ == "__main__":
    test_api_connection()