# CPU 기반 Dockerfile
FROM ubuntu:22.04

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:${PATH}"

# 시스템 라이브러리 경로를 LD_LIBRARY_PATH에 추가 (llama-cpp-python 빌드 시 필요)
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# 기본 패키지 설치
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    libgomp1 \
    libopenblas-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    pkg-config \
    cmake \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

# TOS 동의
RUN /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    /opt/conda/bin/conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# pip 최신화
RUN python -m pip install --upgrade pip

WORKDIR /workspace

# Python 의존성 설치 (다운로드 이어받기 & 재시도)
RUN /opt/conda/bin/conda create -y -n chat python=3.11
# conda 환경 활성화 및 pip 최신화
RUN /opt/conda/bin/conda run -n chat python -m pip install --upgrade pip

# conda 환경에서 pip로 llama-cpp-python 설치
# RUN /opt/conda/bin/conda run -n chat pip install --upgrade --quiet llama-cpp-python

# (선택) conda-forge에서 llama-cpp-python 설치 예시 (pip 대신 사용 가능)
RUN /opt/conda/bin/conda install -n chat -c conda-forge llama-cpp-python

# (선택) llama-cpp-python 버전 명시 예시 (requirements.txt 또는 pip install 시)
# RUN /opt/conda/bin/conda run -n chat pip install --upgrade --quiet llama-cpp-python==0.2.24

COPY requirements-base.txt .
COPY requirements-ml.txt .

# 1단계: 기본 패키지 설치
RUN /opt/conda/bin/conda run -n chat pip install --no-cache-dir -r requirements-base.txt

# 2단계: ML/대용량 패키지 설치 
RUN /opt/conda/bin/conda run -n chat pip install --no-cache-dir -r requirements-ml.txt

# 소스 코드 복사
COPY . .

# 포트 노출 및 실행 커맨드
EXPOSE 8888
# 이후 실행도 conda 환경에서!
CMD ["/opt/conda/bin/conda", "run", "-n", "chat", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]