#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# start.sh
# - 컨테이너 기동 시 Ollama API 서버를 실행하고,
#   대상 모델(ko-llama-8B)이 없으면 Modelfile로 생성한 뒤
#   서버 프로세스를 포그라운드로 유지합니다.
#
# 사용 전제
# - Modelfile 경로: /root/.ollama/Modelfile/ko-llama-8B
# - 내부 통신 포트: 11434 (localhost 기준)
# - 이미지에 Modelfile이 COPY되어 있어야 함
# -----------------------------------------------------------------------------

# 엄격 모드
# -e: 명령 실패 시 즉시 종료
# -u: 정의되지 않은 변수 사용 시 에러
# -o pipefail: 파이프라인 중간 실패도 감지
set -euo pipefail

# 1) Ollama API 서버 백그라운드 실행
#  - & 로 백그라운드 실행하고, $! 에 PID 저장
ollama serve & PID=$!

# 2) API 준비 대기
#  - /api/tags 엔드포인트로 서버 준비 상태 폴링
#  - 성공할 때까지 2초 간격으로 재시도
until ollama list >/dev/null 2>&1; do
  echo "[start.sh] Waiting for Ollama API on :11434 ..."
  sleep 2
done
# 3) 모델 존재 확인 후, 없으면 Modelfile로 생성
#  - ollama list 출력에서 정확히 'ko-llama-8B'가 있는지 검사
#  - -f 인자는 '파일 경로'여야 하므로 디렉터리가 아닌 실제 Modelfile 지정
if ! ollama list | grep -q "^ko-llama-8B"; then
  echo "[start.sh] Model 'ko-llama-8B' not found. Creating from Modelfile..."
  ollama create ko-llama-8B -f /root/.ollama/Modelfile/ko-llama-8B
fi

# 4) 모델 등록 완료 대기
#  - 생성/등록이 완료될 때까지 1초 간격 폴링
until ollama list | grep -q "^ko-llama-8B"; do
  echo "[start.sh] Waiting for model registration 'ko-llama-8B' ..."
  sleep 1
done
echo "[start.sh] Model 'ko-llama-8B' is ready."

# 5) 서버 프로세스를 포그라운드로 유지
#  - 컨테이너의 메인 프로세스로 ollama serve 를 유지
wait $PID