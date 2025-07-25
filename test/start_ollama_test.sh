#!/bin/bash

echo "🚀 Ollama 챗봇 테스트 환경을 시작합니다..."

# Docker Compose로 서비스 시작
echo "📦 Docker 서비스 시작 중..."
docker-compose -f ../test-compose.yaml up -d

# 서비스 상태 확인
echo "🔍 서비스 상태 확인 중..."
docker-compose -f ../test-compose.yaml ps

# Ollama 서비스가 준비될 때까지 대기
echo "⏳ Ollama 서비스 준비 대기 중..."
sleep 10

# Jupyter Lab 접속 정보 출력
echo ""
echo "🎉 환경이 준비되었습니다!"
echo ""
echo "📊 서비스 정보:"
echo "  - Jupyter Lab: http://localhost:8888"
echo "  - Ollama API: http://localhost:11434"
echo ""
echo "🔑 Jupyter Lab 접속:"
echo "  - 토큰: localdev"
echo ""
echo "📝 사용 방법:"
echo "  1. Jupyter Lab에 접속"
echo "  2. ollama_chat_test.ipynb 노트북 열기"
echo "  3. 셀을 순서대로 실행하여 Ollama 테스트"
echo ""
echo "🛑 환경 중지: docker-compose -f ../test-compose.yaml down" 