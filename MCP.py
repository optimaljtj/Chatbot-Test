# mcp 라이브러리 설치 필요: pip install mcp
from mcp.server.fastmcp import FastMCP
import datetime

# 1. MCP 서버 초기화 (서버 이름 정의)
mcp = FastMCP("Company-Internal-Server")

# 2. Tool 예시: AI가 외부 기능을 실행 (서버 상태 확인)
@mcp.tool()
def get_server_status(server_name: str) -> str:
    """특정 서버의 상태를 확인합니다. (예: 'prod-01')"""
    # 실제로는 여기서 서버 API를 호출하는 로직이 들어갑니다.
    return f"서버 {server_name}은 현재 '정상' 작동 중입니다. (부하: 15%)"

# 3. Resource 예시: AI가 데이터를 읽어감 (로그 파일 읽기)
@mcp.resource("logs://system-errors")
def get_recent_errors() -> str:
    """최근 시스템 에러 로그를 반환합니다."""
    return "[2026-06-03 10:00:00] 접속 지연 발생\n[2026-06-03 10:05:00] DB 연결 끊김"

# 4. Prompt 예시: AI에게 작업을 시키는 표준 양식
@mcp.prompt()
def summarize_incident(incident_id: str) -> str:
    """장애 발생 시 일관된 형식으로 요약 보고서를 작성합니다."""
    return f"""
    당신은 시스템 운영자입니다. 다음 장애 ID({incident_id})에 대해:
    1. 무엇이 문제인가?
    2. 데이터 리소스(logs://system-errors)를 참조하여 분석하세요.
    3. [보고서] 형식으로 정리하세요.
    """

# 5. 서버 실행
if __name__ == "__main__":
    mcp.run()