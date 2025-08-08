"""모듈 엔트리 포인트.

`python -m rag_chatbot.backend` 로 CLI를 실행합니다.
"""

from rag_chatbot.backend.cli import app


if __name__ == "__main__":
    app()


