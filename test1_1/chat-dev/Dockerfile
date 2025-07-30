FROM ubuntu:22.04

# ----- 환경변수 설정 -----
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONPATH="/home/jovyan/work:${PYTHONPATH}"

# ----- 시스템 패키지 업데이트 및 필수 패키지 설치 -----
RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-pip \
        python3-venv \
        wget \
        git \
        build-essential \
        curl \
        iputils-ping \
        && apt-get clean && rm -rf /var/lib/apt/lists/*

# ----- 가상환경 생성 및 활성화 -----
RUN python3 -m venv /opt/venv

# ----- pip 업그레이드 및 기본 패키지 설치 -----
RUN /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install --no-cache-dir \
    jupyter \
    notebook \
    jupyterlab \
    ipykernel

# ----- 프로젝트 requirements 파일 복사 및 설치 -----
COPY requirements-base.txt /tmp/
COPY requirements-ml.txt /tmp/

# PyTorch를 먼저 설치 (타임아웃 방지)
RUN /opt/venv/bin/pip install --no-cache-dir --timeout 3000 torch 

# 기본 requirements 설치
RUN /opt/venv/bin/pip install --no-cache-dir --timeout 3000 -r /tmp/requirements-base.txt

# ML requirements 설치 (PyTorch는 이미 설치됨)
RUN /opt/venv/bin/pip install --no-cache-dir --timeout 3000 -r /tmp/requirements-ml.txt

# ----- ipykernel을 통해 커널 등록 (가상환경) -----
RUN /opt/venv/bin/python -m ipykernel install --user --name=venv --display-name "Python (venv)"

# ----- 작업 디렉터리 설정 (볼륨 마운트와 일치) -----
WORKDIR /home/jovyan/work

# ----- JupyterLab 실행 (가상환경 사용) -----
CMD ["/opt/venv/bin/jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=devtoken", \
     "--notebook-dir=/home/jovyan/work"] 
