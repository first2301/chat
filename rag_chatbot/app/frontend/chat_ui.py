import gradio as gr
import requests
import time

BACKEND_QUERY_URL = "http://backend:8000/rag/query"


def chat(message, history):
    # OpenAI 스타일 입력(message는 문자열 또는 {role, content} dict)
    content = message.get("content") if isinstance(message, dict) else message
    try:
        resp = requests.post(
            BACKEND_QUERY_URL,
            json={"question": content},
        )
        data = resp.json()
        answer = data.get("answer", "")
    except Exception as e:
        answer = f"오류가 발생했습니다: {e}"

    # Gradio 제너레이터를 사용한 스트리밍(의사-스트리밍)
    if not answer:
        yield ""
        return

    step = 40  # 청크 크기(문자 수)
    for i in range(0, len(answer), step):
        partial = answer[: i + step]
        yield partial
        time.sleep(0.01)


app = gr.ChatInterface(
    fn=chat,  # 채팅 함수
    type="messages",  # OpenAI 스타일 메시지 포맷 사용
    submit_btn="전송",  # 전송 버튼 텍스트
    css="footer{display:none!important}",  # 푸터 숨김 CSS
)

if __name__ == "__main__":
    app.launch(show_api=False)  # API 문서 표시 비활성화