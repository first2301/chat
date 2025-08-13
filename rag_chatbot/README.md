## RAG Chatbot 시스템 개요

본 문서는 `rag_chatbot` 경로 기준으로 RAG 시스템의 핵심 구성과 운영 방법을 중요도 및 참고 순서에 따라 정리합니다. 빠르게 기동하고, 문제를 진단·해결하는 데 필요한 정보부터 제공합니다.

### 1) 빠른 시작(우선 확인)
- **사전 요구**
  - Docker Desktop + WSL2(Windows) 또는 네이티브 Docker(리눅스/맥)
  - 포트: 호스트 11435→컨테이너 11434(Ollama), 6333(Qdrant), 8000(백엔드)
- **기동**
  - 프로젝트 루트에서 실행:
```bash
docker compose -f rag_chatbot/rnd-rag-compose.yaml up -d --build
```
- **상태 점검**
```bash
curl http://localhost:8000/healthz
# 정상 예: {"vector_store_loaded": true, "embedding_model_name": "...", "ollama_model": "...", "ollama_base_url": "http://ollama:11434"}
```
- **테스트 질의(LLM-only)**
```bash
curl -X POST "http://localhost:8000/rag/test_query/ ㅇㅇㅇ 회사 소개"
```

### 2) 서비스 구성(중요)
- **`ollama`**: LLM 추론 서버. 내부 포트 11434로 수신. 백엔드는 `http://ollama:11434`로 접속.
- **`qdrant`**: 벡터DB. 6333(HTTP)/6334(gRPC). 기본 컬렉션 `rag_collection`.
- **`rag-backend`**: FastAPI API 서버. 애플리케이션 수명 주기에서 RAGAgent 초기화.

구성 파일: `rag_chatbot/rnd-rag-compose.yaml`

필수 환경변수(컴포즈에서 주입):
- `OLLAMA_BASE_URL=http://ollama:11434`
- `OLLAMA_MODEL=ko-llama-8B`
- `QDRANT_URL=http://qdrant:6333`
- `QDRANT_COLLECTION=rag_collection`
- `EMBEDDING_MODEL_NAME=/app/embedding_models/BGE-m3-ko` (또는 허깅페이스 모델명)
- `DATA_DIR=/app/data`

### 3) 설정/환경 변수 로직(중요)
- 코드 위치: `app/backend/src/services/rag/config.py`
- 우선순위: 환경변수 > 기본값. 예) `OLLAMA_BASE_URL` 미설정 시 기본값은 `http://localhost:11434`이므로, 도커 내부에서는 반드시 환경변수로 덮어써야 합니다.
- 검증: 앱 시작 시 `Config.load_env() → resolve_paths() → validate()` 순으로 필수 키와 경로를 확인합니다.

### 4) 백엔드 구조(핵심 흐름)
- 엔트리포인트: `app/backend/main.py`
- Lifespan: `app/backend/src/api/lifespan.py` → `RAGAgent` 생성 및 `app.state.agent` 저장
- 라우터: `app/backend/src/api/routers`
  - `health.py`: `GET /healthz`
  - `rag.py`: 질의/인덱싱 API
  - DI: `app/backend/src/api/deps.py`에서 `get_agent`

### 5) RAG 내부 동작(매우 중요)
- 클래스: `app/backend/src/services/rag/rag_agent.py`
- 임베딩: `HuggingFaceEmbeddings` (`EMBEDDING_MODEL_NAME`, CPU/CUDA 자동 판단)
- 벡터스토어(우선순위)
  1. Qdrant(`app/backend/src/services/rag/vector_store_qdrant.py`)
     - 컬렉션 없으면 임베딩 차원으로 생성
  2. 폴백: FAISS(로컬 디스크)
- 체인: 수동 컨텍스트 체인 + LCEL 체인(둘 다 제공)
- 자동 인덱싱: `ON_MISSING_VECTOR_STORE=auto_build`일 때 `DATA_DIR` 스캔으로 초기 업서트 가능

### 6) 데이터 인덱싱
- 데이터 경로: 컨테이너 기준 `DATA_DIR=/app/data`
- 글롭: `DOC_GLOBS`(기본 `**/*.pdf,**/*.txt`)
- API:
  - `POST /rag/index/files`: 파일 경로 목록 제공 또는 `DATA_DIR` 자동 스캔
  - `POST /rag/index/urls`: URL 목록 인덱싱
  - `POST /rag/index/text`: 텍스트 직접 업서트(간단 처리)

### 7) API 사용(실전)
- 헬스체크
```bash
curl http://localhost:8000/healthz
```
- LLM-only 테스트(벡터스토어 불필요)
```bash
curl -X POST "http://localhost:8000/rag/test_query/회사 소개 부탁해"
```
- RAG 질의
```bash
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "문서 요약 부탁해", "mode": "lcel"}'
```
- 파일 인덱싱
```bash
curl -X POST http://localhost:8000/rag/index/files \
  -H "Content-Type: application/json" \
  -d '{"paths": ["/app/data/sample.pdf"], "force_rebuild": false}'
```

### 8) Qdrant 운용 팁
- 임베딩 차원 불일치 시 에러 → 새 컬렉션명 사용 또는 기존 컬렉션 삭제 후 재생성
```bash
curl -X DELETE http://localhost:6333/collections/rag_collection
# 또는 환경변수로
# QDRANT_COLLECTION=rag_collection_v2
```

### 9) 트러블슈팅(빈도순)
- 백엔드 500 + `httpx.ConnectTimeout` (Ollama)
  - 원인: Ollama 미리스닝/지연, 잘못된 URL(로컬호스트 기본값), 네트워크 대기 부족
  - 조치: `OLLAMA_BASE_URL=http://ollama:11434` 확인, Ollama 로그/리스닝 확인, 헬스체크/재시도
- Qdrant 연결/차원 이슈
  - 컬렉션 차원 확인 후 불일치 시 새 컬렉션 사용 또는 삭제
- 코드/설정 미반영(캐시)
  - 강제 재생성: `--no-cache`/`--force-recreate`
- 윈도우/WSL 파일 동기화 지연
  - 컨테이너 재시작으로 반영 보장

### 10) 운영 체크리스트
- compose 기동:
```bash
docker compose -f rag_chatbot/rnd-rag-compose.yaml up -d --build
```
- 서비스 상태:
```bash
docker compose -f rag_chatbot/rnd-rag-compose.yaml ps
```
- Ollama 준비 확인(백엔드 컨테이너 내부):
```bash
docker exec -it rag-backend sh -lc "curl -sS http://ollama:11434/api/tags | head -c 200"
```

### 11) 개발 모드 팁
- 특정 서비스만 재시작:
```bash
docker compose -f rag_chatbot/rnd-rag-compose.yaml up -d --no-deps --force-recreate backend
```
- 빌드 캐시 무시:
```bash
docker compose -f rag_chatbot/rnd-rag-compose.yaml build --no-cache
```
- 로컬 단독 실행 시(비권장): `OLLAMA_BASE_URL=http://localhost:11434`로 설정

### 12) 디렉터리 참고
- 백엔드: `app/backend/`
  - API: `src/api/routers/*`, DI: `src/api/deps.py`, Lifespan: `src/api/lifespan.py`
  - RAG: `src/services/rag/*` (에이전트, 설정, 인덱션, Qdrant)
- Ollama 데이터/모델: `app/ollama_server/`, `app/ai_models/`
- Qdrant 스토리지/스냅샷: `app/vector_data/qdrant_storage/`, `app/vector_data/snapshot_backup/`

### 13) 보안/비밀정보
- `QDRANT_API_KEY` 사용 시 `.env`/시크릿으로 관리. 현재 예시는 로컬 개발 전제(키 미사용).

---
문의/개선 제안은 본 리포지토리 이슈로 등록해 주세요. 운영 환경에서는 헬스체크와 재시도 정책(백오프)을 강화하는 것을 권장합니다.


