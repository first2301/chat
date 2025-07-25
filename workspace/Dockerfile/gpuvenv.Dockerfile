FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

#––[1] 환경 변수 설정 –––––––––––––––––––––––––––––––––––––––––––
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
# venv 위치 (/opt/venv/bin)을 PATH 맨 앞에 두어 pip‧python이 기본 해석기로 잡히도록
ENV PATH="/opt/venv/bin:${PATH}"

#––[2] 시스템 패키지 설치 –––––––––––––––––––––––––––––––––––––––
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        wget curl git build-essential cmake pkg-config \
        python3 python3-pip python3-venv python3-dev \
        libgomp1 libopenblas-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*

#––[3] 가상환경 생성 & pip 최신화 –––––––––––––––––––––
RUN python3 -m venv /opt/venv && \
    pip install --upgrade pip

#––[4] 파이썬 의존성 설치 –––––––––––––––––––––––––––––––––––––––
WORKDIR /workspace

# requirements 파일 복사
COPY requirements-base.txt requirements-ml.txt ./

# 1단계: 공통 패키지
RUN pip install --no-cache-dir -r requirements-base.txt

# 2단계: ML/대용량 패키지

RUN pip install --no-cache-dir -r requirements-ml.txt

#––[5] llama-cpp-python (CUDA 지원) 설치 ––––––––––––––––––––––––
# RUN CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
#     pip install --no-cache-dir llama-cpp-python

#––[6] 애플리케이션 소스 복사 –––––––––––––––––––––––––––––––––––
COPY . .

#––[7] 포트 노출 및 컨테이너 시작 커맨드 ––––––––––––––––––––––
EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", \
     "--NotebookApp.token=", "--NotebookApp.password="]
