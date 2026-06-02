import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END
from typing import TypedDict
from neo4j import GraphDatabase

# --------------------------------------------------------
# 0. 환경 변수 로드 및 페이지 설정
# --------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="Gemini GraphRAG Agent",
    page_icon="🕸️",
    layout="wide"
)

st.title("🕸️ 제미나이 × LangGraph × Neo4j 에이전트 챗봇")
st.caption("Graph RAG 메커니즘을 검증하기 위한 완벽한 데모 프로그램입니다.")

# --------------------------------------------------------
# 1. 인프라 연결 초기화 (Gemini & Neo4j)
# --------------------------------------------------------
@st.cache_resource
def init_connections():
    # 최신 google-genai 클라이언트 초기화 (2026년 표준 SDK)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("⚠️ GEMINI_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
        st.stop()
        
    client = genai.Client(api_key=api_key)
    
    # Neo4j 그래프 DB 드라이버 초기화 (Aura 또는 Sandbox 연결용)
    uri = os.getenv("NEO4J_URI")
    username = os.getenv("NEO4J_USERNAME", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    if not uri or not password:
        st.error("⚠️ NEO4J_URI 또는 NEO4J_PASSWORD가 설정되지 않았습니다. .env 파일을 확인하세요.")
        st.stop()
        
    neo4j_driver = GraphDatabase.driver(uri, auth=(username, password))
    return client, neo4j_driver

ai_client, graph_db = init_connections()

# --------------------------------------------------------
# 2. 에이전트 전용 툴(Tool) 정의: 지식 그래프 조회
# --------------------------------------------------------
def query_knowledge_graph(search_keyword: str) -> str:
    """지식 그래프(Neo4j)에서 핵심 키워드와 관련된 노드(개체) 및 연결 관계를 조회합니다."""
    query = """
    MATCH (n)-[r]->(m)
    WHERE n.name CONTAINS $keyword OR m.name CONTAINS $keyword OR type(r) CONTAINS $keyword
    RETURN n.name AS source, type(r) AS relation, m.name AS target
    LIMIT 15
    """
    try:
        with graph_db.session() as session:
            result = session.run(query, keyword=search_keyword)
            records = [f"({row['source']})-[{row['relation']}]->({row['target']})" for row in result]
            
            if not records:
                return f"🔍 그래프 DB 내에서 '{search_keyword}'와 관련된 의미 있는 지식 연결망을 찾지 못했습니다."
            return "\n".join(records)
    except Exception as e:
        return f"❌ 그래프 DB 조회 중 하드웨어/네트워크 오류 발생: {str(e)}"

# LLM이 매핑해서 실행할 수 있도록 딕셔너리로 등록
tools_map = {"query_knowledge_graph": query_knowledge_graph}

# --------------------------------------------------------
# 3. LangGraph 기반 에이전트 상태 및 워크플로우 모델링
# --------------------------------------------------------
class AgentState(TypedDict):
    messages: list   # 누적 대화 메시지 목록
    next_step: str  # 다음 노드로 가기 위한 흐름 제어 태그

def call_gemini_agent(state: AgentState):
    """현재까지의 대화 맥락을 제미나이에게 전달하고 다음 행동을 결정받습니다."""
    messages = state["messages"]
    
    # 제미나이 SDK 규격에 맞는 Function(Tool) 선언
    tool_def = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="query_knowledge_graph",
                description="특정 시스템, 부품, 인물, 프로젝트 간의 연관 관계나 사내 지식이 필요할 때 Neo4j DB를 조회합니다.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "search_keyword": types.Schema(
                            type=types.Type.STRING,
                            description="지식 그래프에서 검색할 단일 핵심 키워드 (예: 장비명, 프로젝트명)"
                        )
                    },
                    required=["search_keyword"]
                )
            )
        ]
    )
    
    # 대화 이력에서 텍스트 콘텐츠 추출
    contents = [msg["content"] for msg in messages]
    
    # 시스템 명령 지침 생성
    config = types.GenerateContentConfig(
        tools=[tool_def],
        system_instruction=(
            "당신은 사내 지식 그래프(Neo4j)를 보조 기억 장치로 사용하는 전문 AI 에이전트입니다. "
            "단순한 일상 대화는 즉시 친절하게 답하고, 특정 전문 지식이나 관계 분석이 필요한 질문이 들어오면 "
            "반드시 'query_knowledge_graph' 도구를 호출하여 팩트를 검증한 뒤 답변하십시오."
        )
    )
    
    # gemini-2.5-flash 모델 호출
    response = ai_client.models.generate_content(
        model='gemini-2.5-flash',
        contents=contents,
        config=config
    )
    
    # 제미나이가 도구(Tool) 호출을 요청했는지 검사
    if response.function_calls:
        call = response.function_calls[0]
        return {
            "messages": messages + [{
                "role": "assistant", 
                "content": f"⚙️ [에이전트 판단] 그래프 DB 조회가 필요합니다. 키워드: '{call.args['search_keyword']}'", 
                "function_call": call
            }],
            "next_step": "call_tool"
        }
    else:
        return {
            "messages": messages + [{"role": "assistant", "content": response.text}],
            "next_step": "end"
        }

def execute_tool(state: AgentState):
    """제미나이가 요청한 도구를 실행하고 그 결과를 컨텍스트에 추가하여 되돌려줍니다."""
    messages = state["messages"]
    last_msg = messages[-1]
    call = last_msg["function_call"]
    
    # 실제 등록된 파이썬 함수 동적 실행
    tool_result = tools_map[call.name](**call.args)
    
    # DB 조회 결과를 다음 질문의 컨텍스트로 주입
    enriched_content = (
        f"[그래프 DB 실시간 조회 결과]\n{tool_result}\n\n"
        f"위의 신뢰할 수 있는 그래프 지식 데이터를 기반으로 사용자의 질문에 정확하고 왜곡 없는 최종 답변을 작성해 주세요."
    )
    
    return {
        "messages": messages + [{"role": "user", "content": enriched_content}],
        "next_step": "call_agent"
    }

def route_next(state: AgentState):
    """조건부 라우팅 제어 함수"""
    return state["next_step"]

# LangGraph 워크플로우 맵 빌드 및 컴파일
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_gemini_agent)
workflow.add_node("tool", execute_tool)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    route_next,
    {
        "call_tool": "tool",
        "end": END
    }
)
workflow.add_edge("tool", "agent")
app_agent = workflow.compile()

# --------------------------------------------------------
# 4. Streamlit 기반 사용자 인터페이스(UI) 구성
# --------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 기존 대화 이력 렌더링 (시스템 백그라운드 로직용 메시지는 제외하고 노출)
for msg in st.session_state.chat_history:
    if not msg["content"].startswith("[그래프 DB"):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# 사용자 메시지 입력창
if user_input := st.chat_input("에이전트에게 지식 기반 질문을 입력하세요..."):
    with st.chat_message("user"):
        st.markdown(user_input)
        
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # 에이전트 추론 루프 가동
    with st.chat_message("assistant"):
        with st.spinner("에이전트가 상태 그래프 흐름에 따라 사고하는 중..."):
            initial_state = {"messages": st.session_state.chat_history, "next_step": ""}
            final_output = app_agent.invoke(initial_state)
            
            # 그래프 연산이 완료된 전체 히스토리 반영
            st.session_state.chat_history = final_output["messages"]
            
            # 에이전트의 최종 정제 답변 출력
            st.markdown(st.session_state.chat_history[-1]["content"])