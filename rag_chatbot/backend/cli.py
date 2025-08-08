"""RAG Chatbot Backend용 Typer 기반 CLI.

- init: 문서 경로(파일/URL/PDF)로 벡터스토어를 생성하고 저장
- load: .env의 VECTOR_STORE_PATH로 벡터스토어 로드 검증
- q: 질의 처리(로컬 에이전트 또는 원격 FastAPI)
- health: 헬스 체크(로컬/원격)
- serve: Uvicorn으로 FastAPI 서버 기동

사용 예시:
    python -m rag_chatbot.backend init -d data/doc.pdf -d https://example.com
    python -m rag_chatbot.backend q "질문 내용" -m lcel
    python -m rag_chatbot.backend health --remote http://localhost:8000
    python -m rag_chatbot.backend serve --host 0.0.0.0 --port 8000
"""

from typing import List, Optional

import typer
import requests
from dotenv import load_dotenv
import uvicorn

from rag_chatbot.backend.src.services.rag.rag_agent import RAGAgent
from rag_chatbot.backend.src.services.rag.config import Config


app = typer.Typer(help="RAG Chatbot Backend CLI")


@app.command()
def init(
    docs: List[str] = typer.Option(
        None,
        "-d",
        "--docs",
        help="인덱싱할 문서 경로(파일/URL/PDF). 옵션을 여러 번 지정하여 여러 문서를 전달",
    )
):
    """문서로부터 벡터스토어를 생성하고 저장합니다.

    - .env의 VECTOR_STORE_PATH가 필요합니다.
    - PDF는 PyMuPDFLoader로 처리됩니다.
    """
    load_dotenv()
    if not docs:
        typer.echo("[오류] --docs 옵션으로 하나 이상의 문서를 지정하세요.")
        raise typer.Exit(code=1)
    if not Config.vector_store_path:
        typer.echo("[오류] VECTOR_STORE_PATH 환경변수가 설정되어 있지 않습니다.")
        raise typer.Exit(code=1)

    try:
        agent = RAGAgent(document_paths=docs)
        # _create_vector_store_from_documents는 생성자에서 호출됨
        agent.save_vector_store()
        typer.echo(f"[완료] 벡터스토어 저장 경로: {Config.vector_store_path}")
    except Exception as exc:
        typer.echo(f"[오류] 인덱싱 실패: {exc}")
        raise typer.Exit(code=2)


@app.command()
def load():
    """.env의 VECTOR_STORE_PATH로 벡터스토어 로드를 검증합니다."""
    load_dotenv()
    if not Config.vector_store_path:
        typer.echo("[오류] VECTOR_STORE_PATH 환경변수가 설정되어 있지 않습니다.")
        raise typer.Exit(code=1)
    try:
        agent = RAGAgent()
        agent.load_vector_store()
        typer.echo("[완료] 벡터스토어 로드 성공")
    except Exception as exc:
        typer.echo(f"[오류] 로드 실패: {exc}")
        raise typer.Exit(code=2)


@app.command(name="q")
def query(
    question: str = typer.Argument(..., help="질문 텍스트"),
    mode: str = typer.Option("lcel", "-m", "--mode", help="manual | lcel"),
    remote: Optional[str] = typer.Option(None, "--remote", help="원격 FastAPI URL (예: http://localhost:8000)"),
):
    """RAG 질의를 수행합니다.

    - 기본은 로컬 에이전트로 처리합니다.
    - --remote 지정 시 FastAPI 서버의 /rag/query를 호출합니다.
    """
    load_dotenv()
    if remote:
        try:
            url = remote.rstrip("/") + "/rag/query"
            res = requests.post(url, json={"question": question, "mode": mode}, timeout=60)
            res.raise_for_status()
            data = res.json()
            typer.echo(data.get("answer") or data)
        except Exception as exc:
            typer.echo(f"[오류] 원격 요청 실패: {exc}")
            raise typer.Exit(code=2)
        return

    # 로컬 처리
    try:
        agent = RAGAgent()
        agent.load_vector_store()
        if mode == "manual":
            answer = agent.query(question)
        else:
            answer = agent.query_lcel(question)
        typer.echo(str(answer))
    except Exception as exc:
        typer.echo(f"[오류] 로컬 질의 실패: {exc}")
        raise typer.Exit(code=2)


@app.command()
def health(
    remote: Optional[str] = typer.Option(None, "--remote", help="원격 FastAPI URL (예: http://localhost:8000)"),
):
    """서비스 헬스 상태를 확인합니다.

    - --remote 지정 시 /healthz 호출 결과를 출력합니다.
    - 지정하지 않으면 로컬 설정 요약을 출력합니다.
    """
    load_dotenv()
    if remote:
        try:
            url = remote.rstrip("/") + "/healthz"
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            typer.echo(res.json())
        except Exception as exc:
            typer.echo(f"[오류] 원격 헬스 체크 실패: {exc}")
            raise typer.Exit(code=2)
        return

    # 로컬 설정 요약
    summary = {
        "embedding_model_name": Config.embedding_model_name,
        "ollama_model": Config.ollama_model_name,
        "ollama_base_url": Config.ollama_base_url,
        "vector_store_path": Config.vector_store_path,
    }
    typer.echo(summary)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="바인딩 호스트"),
    port: int = typer.Option(8000, "--port", help="바인딩 포트"),
    reload: bool = typer.Option(False, "--reload", help="개발 모드 자동 리로드"),
):
    """Uvicorn으로 FastAPI 서버를 실행합니다."""
    uvicorn.run("rag_chatbot.backend.main:app", host=host, port=port, reload=reload)


