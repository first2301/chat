import gradio as gr
import requests

BACKEND_QUERY_URL = "http://127.0.0.1:8000/query"


def chat(message, history):
    # OpenAI 스타일 입력(message는 문자열 또는 {role, content} dict)
    content = message.get("content") if isinstance(message, dict) else message
    try:
        resp = requests.post(
            BACKEND_QUERY_URL,
            json={"role": "user", "content": content},
            timeout=60,
        )
        # resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer", "")
    except Exception as e:
        answer = f"오류가 발생했습니다: {e}"
    return {"role": "assistant", "content": answer}


app = gr.ChatInterface(
    fn=chat,  # 채팅 함수
    type="messages",  # OpenAI 스타일 메시지 포맷 사용
    submit_btn="전송",  # 전송 버튼 텍스트
    css="footer{display:none!important}",  # 푸터 숨김 CSS
)

if __name__ == "__main__":
    app.launch(show_api=False)  # API 문서 표시 비활성화