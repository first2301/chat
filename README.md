# Chat Development Environment

이 프로젝트는 Jupyter Notebook 기반의 개발 환경을 Docker로 구성한 것입니다.

## 구성 요소

- **Backend**: Python 기반 백엔드 서비스
- **Frontend**: React 기반 프론트엔드 애플리케이션
- **R&D Environment**: Jupyter Notebook 기반 연구 개발 환경

## 빠른 시작

### 1. 환경 변수 설정

`test` 디렉토리에 `.env` 파일을 생성하고 다음 내용을 추가하세요:

```bash
# Jupyter 설정
JUPYTER_TOKEN=your_secure_token_here
JUPYTER_PASSWORD=your_password_here

# 개발 환경 설정
NODE_ENV=development
DEBUG=true
```

### 2. Jupyter Notebook 환경 실행

```bash
# 테스트 환경 실행
docker compose -f test-compose.yaml up -d

# 로그 확인
docker compose -f test-compose.yaml logs -f notebook

# 환경 중지
docker compose -f test-compose.yaml down
```

### 3. Jupyter Lab 접속

브라우저에서 `http://localhost:8888`로 접속하세요.

## 주요 개선사항

### 보안 강화
- 환경 변수를 통한 토큰/비밀번호 관리
- 읽기 전용 볼륨 마운트
- 보안 옵션 추가

### 네트워크 설정
- 별도 네트워크 구성
- 서브넷 설정으로 격리

### 리소스 관리
- CPU, 메모리 제한
- 프로세스 수 제한
- 공유 메모리 크기 증가

### 모니터링
- 헬스체크 추가
- 자동 재시작 설정

## 디렉토리 구조

```
chat/
├── backend/          # 백엔드 서비스
├── frontend/         # 프론트엔드 애플리케이션
├── test/             # Jupyter 개발 환경
│   ├── Dockerfile
│   ├── requirements-base.txt
│   ├── requirements-ml.txt
│   └── test.ipynb
├── rnd-chat-dev/     # 연구 개발 환경
├── docker-compose.yaml
└── test-compose.yaml
```

## 문제 해결

### 컨테이너가 시작되지 않는 경우
```bash
# 로그 확인
docker compose -f test-compose.yaml logs notebook

# 컨테이너 재빌드
docker compose -f test-compose.yaml build --no-cache
```

### permission 권한 문제 해결
sudo usermod -aG docker $USER


### 메모리 부족 오류
`test-compose.yaml`의 `memory` 제한을 조정하세요.

## 참고 자료

- [Docker Compose GPU 지원](https://docs.docker.com/compose/how-tos/gpu-support/)
- [Jupyter Lab 설정](https://jupyterlab.readthedocs.io/en/stable/)