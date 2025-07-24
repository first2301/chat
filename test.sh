#!/bin/bash
# -----------------------------
# 코로케이션 GPU 서버 자원 모니터링 스크립트
# 작성자: 내부 관리용
# -----------------------------

echo "===== [ GPU 상태 (nvidia-smi) ] ====="
nvidia-smi

echo
echo "===== [ GPU 프로세스 모니터링 (pmon) ] ====="
nvidia-smi pmon -c 1

echo
echo "===== [ CPU/메모리 사용 현황 (top 5) ] ====="
ps aux --sort=-%mem | head -n 6

echo
echo "===== [ 전체 메모리 현황 ] ====="
free -h

echo
echo "===== [ 공유 메모리(/dev/shm) 사용량 ] ====="
df -h /dev/shm

echo
echo "===== [ 컨테이너별 자원 사용 현황 ] ====="
docker stats --no-stream

echo
echo "===== [ 디스크 사용량 ] ====="
df -h

echo
echo "===== [ 네트워크 인터페이스 상태 ] ====="
netstat -i

echo
echo "===== [ 디스크 I/O 모니터링 (최근 3초 평균) ] ====="
iostat 1 3 | tail -n 10
