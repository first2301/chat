# backend/Dockerfile
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu22.04

# 1) 시스템 의존성 및 Python 설치
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      python3 python3-pip build-essential && \
    rm -rf /var/lib/apt/lists/*

# 2) 작업 디렉토리 설정
WORKDIR /app

# 3) Python 의존성 설치 (타임아웃 및 재시도 설정)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 4) 애플리케이션 소스 코드 복사
# 현재 구조에서는 backend/ 바로 아래에 소스 코드가 있다고 가정
COPY . .

# 5) 포트 노출
EXPOSE 8000

# 6) 컨테이너 시작 시 FastAPI 서버 기동
# main.py가 backend/ 바로 아래에 있다고 가정
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
