# src/convpool/main.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from gradio.routes import mount_gradio_app
import gradio as gr

# --- 아주 단순한 블록을 직접 만들어서 mount (import 문제 배제) ---
def build_sender_blocks():
    with gr.Blocks(title="Sender") as app:
        gr.Markdown("## 📨 Sender\n여기에 그리기 UI가 들어올 예정입니다.")
    return app

def build_receiver_blocks():
    with gr.Blocks(title="Receiver") as app:
        gr.Markdown("## 📺 Receiver\n여기에 영상/그래프가 뜰 예정입니다.")
    return app

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

# 루트 -> /sender 리다이렉트 (문서 숨김)
@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/sender")

# Gradio 앱 mount
sender = build_sender_blocks()
receiver = build_receiver_blocks()
mount_gradio_app(app, sender, path="/sender")
mount_gradio_app(app, receiver, path="/receiver")

# 디버그용 라우트 목록
@app.get("/routes")
def routes():
    return [{"path": r.path, "name": getattr(r, "name", None)} for r in app.routes]
