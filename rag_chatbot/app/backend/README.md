### RAG Chatbot Backend 실행 가이드

이 문서는 `rag_chatbot/backend` 기준의 로컬/서버 실행과 CLI 사용 방법을 설명합니다.

### 1) 사전 준비
- Python 3.10+ (권장 3.11)
- 패키지 설치
  ```bash
  pip install fastapi uvicorn typer requests python-dotenv
  pip install langchain langchain-community langchain-ollama langchain-huggingface faiss-cpu pymupdf sentence-transformers
  ```
- `.env` 파일 생성(같은 디렉터리: `rag_chatbot/backend/.env`)
  ```env
  # [필수]
  VECTOR_STORE_PATH=./vector_db

  # [선택]
  EMBEDDING_MODEL_NAME=BAAI/bge-m3
  EMBEDDING_DEVICE=cpu                # gpu 사용 시 cuda
  CHUNK_SIZE=500
  CHUNK_OVERLAP=50
  K=20

  # Ollama 설정
  OLLAMA_BASE_URL=http://localhost:11434
  OLLAMA_MODEL=ko-llama-8B
  ```

### 2) Typer CLI로 빠른 사용(권장)
- 모듈 엔트리 한 줄 명령: `python -m rag_chatbot.backend <subcommand>`

- 인덱스 생성(파일/URL/PDF 혼합 가능)
  ```bash
  python -m rag_chatbot.backend init -d path/to/doc1.pdf -d https://example.com
  ```

- 인덱스 로드 검증
  ```bash
  python -m rag_chatbot.backend load
  ```

- 질의(로컬, 기본 lcel 체인)
  ```bash
  python -m rag_chatbot.backend q "질문 내용" -m lcel
  # 수동 체인
  python -m rag_chatbot.backend q "질문 내용" -m manual
  ```

- 서버 실행(uvicorn)
  ```bash
  python -m rag_chatbot.backend serve --host 0.0.0.0 --port 8000
  ```

- 헬스체크(로컬/원격)
  ```bash
  # 로컬 설정 요약
  python -m rag_chatbot.backend health

  # 원격 FastAPI (예: 로컬 서버)
  python -m rag_chatbot.backend health --remote http://localhost:8000
  ```

- 원격 질의(서버에 요청)
  ```bash
  python -m rag_chatbot.backend q "질문 내용" -m lcel --remote http://localhost:8000
  ```

### 3) FastAPI 엔드포인트
- `GET /healthz`: 상태(벡터스토어 로드 여부, 임베딩/모델/URL)
- `POST /rag/query`: JSON { question, mode: 'manual'|'lcel' }
- `POST /rag/reload`: 벡터스토어 재로딩

### 4) PDF/문서 처리
- `.pdf`는 `PyMuPDFLoader`로 처리됩니다. `pymupdf` 설치가 필요합니다.
- 텍스트/URL은 기본 로더로 수집 후 `RecursiveCharacterTextSplitter`로 분할합니다.

### 5) GPU / CPU 모드
- 로컬 CPU: `.env`에 `EMBEDDING_DEVICE=cpu`
- GPU: `.env`에 `EMBEDDING_DEVICE=cuda` + CUDA 환경 필요
- Ollama도 GPU 컨테이너/서비스에서 모델을 로드하도록 구성하세요.

### 6) 문제 해결
- `ImportError: fastapi/uvicorn`: 상단 패키지 설치 명령으로 해결
- `VECTOR_STORE_PATH 미설정` 오류: `.env`에 경로를 지정하세요(예: `./vector_db`)
- `FAISS load_local` 오류: 호환 옵션이 코드에 포함되어 있으나, 오래된 인덱스는 재생성이 필요할 수 있습니다.

### 7) 디렉터리 구조(요약)
```
rag_chatbot/backend/
  ├─ __main__.py          # python -m rag_chatbot.backend 진입점
  ├─ cli.py               # Typer CLI (init/load/q/health/serve)
  ├─ main.py              # FastAPI 앱 엔트리
  └─ src/
     ├─ api/              # 라우터/DI/수명관리
     └─ services/rag/     # Config/RAGAgent
```

필요 시 Docker 구성도 추가할 수 있습니다. 현재 문서는 로컬 실행을 기준으로 합니다.


