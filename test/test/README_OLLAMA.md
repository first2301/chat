# Ollama 챗봇 테스트 환경

이 환경은 rnd-chat-dev에서 Ollama를 사용한 챗봇 테스트를 위한 것입니다.

## 🚀 빠른 시작

### 1. 환경 시작
```bash
# test 디렉토리에서 실행
./start_ollama_test.sh
```

### 2. Jupyter Lab 접속
- URL: http://localhost:8888
- 토큰: `localdev`

### 3. 테스트 노트북 실행
- `ollama_chat_test.ipynb` 노트북을 열고 셀을 순서대로 실행

## 📋 환경 구성

### 서비스
- **Ollama**: 로컬 LLM 서버 (포트 11434)
- **rnd-chat-dev**: Jupyter Lab 환경 (포트 8888)

### 주요 기능
- ✅ Ollama 서버 연결 테스트
- ✅ 모델 다운로드 및 관리
- ✅ LangChain을 통한 챗봇 생성
- ✅ 한국어 대화 테스트
- ✅ 성능 테스트

## 🔧 사용 가능한 모델

### 기본 모델
- `llama2`: 범용 대화 모델
- `codellama`: 프로그래밍 전문 모델

### 모델 다운로드
```python
# 노트북에서 실행
download_model("llama2")
download_model("codellama")
```

## 📝 테스트 예제

### 1. 기본 대화 테스트
```python
chat_with_bot("안녕하세요! 자기소개를 해주세요.")
```

### 2. 한국어 대화 테스트
```python
system_prompt = "당신은 친근하고 도움이 되는 AI 어시스턴트입니다. 한국어로 대화해주세요."
chat_with_bot("파이썬으로 계산기 프로그램을 만들어주세요.", system_prompt)
```

### 3. 코드 생성 테스트
```python
code_chatbot = create_ollama_chatbot("codellama")
system_prompt = "당신은 프로그래밍 전문가입니다. 코드 예제를 제공해주세요."
chat_with_bot("JavaScript로 배열 정렬 함수를 만들어주세요.", system_prompt)
```

## 🛠️ 문제 해결

### Ollama 서버 연결 실패
1. Docker 서비스 상태 확인:
   ```bash
   docker-compose -f ../test-compose.yaml ps
   ```

2. Ollama 로그 확인:
   ```bash
   docker-compose -f ../test-compose.yaml logs ollama
   ```

### 모델 다운로드 실패
1. 네트워크 연결 확인
2. 충분한 디스크 공간 확보
3. Docker 볼륨 확인:
   ```bash
   docker volume ls | grep ollama
   ```

## 📊 성능 최적화

### 리소스 설정
- **CPU**: 4코어 (rnd-chat-dev), 2코어 (Ollama)
- **메모리**: 4GB (rnd-chat-dev), 8GB (Ollama)

### 성능 테스트
```python
performance_test()
```

## 🛑 환경 정리

### 서비스 중지
```bash
docker-compose -f ../test-compose.yaml down
```

### 볼륨 삭제 (모델 데이터 포함)
```bash
docker-compose -f ../test-compose.yaml down -v
```

## 📚 추가 정보

- [Ollama 공식 문서](https://ollama.ai/docs)
- [LangChain Ollama 통합](https://python.langchain.com/docs/integrations/llms/ollama)
- [Docker Compose 문서](https://docs.docker.com/compose/) 