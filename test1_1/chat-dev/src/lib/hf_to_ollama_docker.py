import os
import requests
import json

def get_ollama_host():
    """환경 변수에서 Ollama 호스트 주소를 가져옵니다."""
    return os.getenv('OLLAMA_BASE_URL', 'http://ollama:11434')

def register_hf_model_to_ollama(model_path, model_name, ollama_host=None):
    """
    HTTP API를 통해 Hugging Face 모델을 ollama에 등록하는 함수입니다.
    Docker 환경에서 사용합니다.
    
    :param model_path: 변환된 모델 파일 경로 (예: 'model.gguf')
    :param model_name: ollama에 등록할 모델 이름 (예: 'my-model')
    :param ollama_host: ollama 서버 주소 (None이면 환경 변수 사용)
    """
    
    if ollama_host is None:
        ollama_host = get_ollama_host()
    
    # 모델 파일 경로 검증
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
        return
    
    print(f"✅ 모델 파일 발견: {model_path}")
    
    # 현재 경로에 있는 Modelfile을 사용
    modelfile_path = os.path.join(os.getcwd(), "Modelfile")
    
    # Modelfile이 존재하는지 확인
    if not os.path.exists(modelfile_path):
        print(f"❌ Modelfile을 찾을 수 없습니다: {modelfile_path}")
        print("현재 디렉토리에 Modelfile이 있는지 확인하세요.")
        return
    
    print(f"✅ Modelfile 발견: {modelfile_path}")
    
    try:
        # Modelfile 내용을 읽어서 API로 전송
        with open(modelfile_path, 'r') as f:
            modelfile_content = f.read()
        
        print(f"Modelfile 내용:\n{modelfile_content}")
        
        # Ollama API를 통해 모델 생성
        response = requests.post(
            f"{ollama_host}/api/create",
            json={
                "name": model_name,
                "modelfile": modelfile_content
            },
            timeout=300  # 5분 타임아웃
        )
        
        if response.status_code == 200:
            print(f"✅ 모델이 성공적으로 등록되었습니다: {model_name}")
            print(response.json())
            
            # 등록된 모델 확인
            check_model_status(model_name, ollama_host)
        else:
            print(f"❌ 모델 등록 실패: {response.status_code}")
            print(response.text)
            print("⚠️ HTTP API 방식이 지원되지 않을 수 있습니다.")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Ollama API 연결 오류: {e}")
        print("⚠️ Docker 네트워크 연결을 확인하세요.")
    except FileNotFoundError as e:
        print(f"❌ 파일을 찾을 수 없습니다: {e}")
    except Exception as e:
        print(f"❌ 예상치 못한 오류: {e}")

def create_modelfile_from_template(model_path, model_name, template_path=None):
    """
    모델 파일을 기반으로 Modelfile을 생성합니다.
    
    :param model_path: 모델 파일 경로
    :param model_name: 모델 이름
    :param template_path: 템플릿 파일 경로 (선택사항)
    """
    
    # 기본 Modelfile 템플릿
    if template_path and os.path.exists(template_path):
        with open(template_path, 'r') as f:
            modelfile_content = f.read()
    else:
        # 기본 템플릿
        modelfile_content = f"""FROM {model_path}
TEMPLATE "{{{{ .System }}}}{{{{ .Prompt }}}}{{{{ .Response }}}}"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
"""
    
    # Modelfile 생성
    modelfile_path = os.path.join(os.getcwd(), "Modelfile")
    with open(modelfile_path, 'w') as f:
        f.write(modelfile_content)
    
    print(f"✅ Modelfile이 생성되었습니다: {modelfile_path}")
    print(f"Modelfile 내용:\n{modelfile_content}")
    
    return modelfile_path

def check_model_status(model_name, ollama_host=None):
    """
    등록된 모델의 상태를 확인합니다.
    """
    if ollama_host is None:
        ollama_host = get_ollama_host()
        
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            for model in models:
                if model['name'] == model_name:
                    print(f"✅ 모델 '{model_name}'이 성공적으로 등록되었습니다.")
                    print(f"모델 크기: {model.get('size', 'N/A')}")
                    return True
            print(f"❌ 모델 '{model_name}'을 찾을 수 없습니다.")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Ollama API 연결 오류: {e}")
        return False

def list_available_models(ollama_host=None):
    """
    사용 가능한 모델 목록을 확인합니다.
    """
    if ollama_host is None:
        ollama_host = get_ollama_host()
        
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=10)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("사용 가능한 모델:")
            for model in models:
                print(f"- {model['name']} (크기: {model.get('size', 'N/A')})")
            return models
        else:
            print(f"API 오류: {response.status_code}")
            return []
    except requests.exceptions.RequestException as e:
        print(f"Ollama API 연결 오류: {e}")
        return []

def test_ollama_connection(ollama_host=None):
    """
    Ollama 서버 연결을 테스트합니다.
    """
    if ollama_host is None:
        ollama_host = get_ollama_host()
        
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=10)
        if response.status_code == 200:
            print("✅ Ollama 서버에 성공적으로 연결되었습니다.")
            return True
        else:
            print(f"❌ Ollama 서버 연결 실패: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Ollama 서버 연결 오류: {e}")
        return False

if __name__ == "__main__":
    # 연결 테스트
    print("Ollama 서버 연결 테스트:")
    test_ollama_connection()
    
    print("\n현재 등록된 모델:")
    list_available_models()
    
    # 사용 예시 (주석 처리)
    # register_hf_model_to_ollama("./models/llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf", "llama-3.2-Korean-Bllossom-3B-gguf-Q4_K_M.gguf")