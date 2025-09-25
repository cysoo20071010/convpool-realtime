# src/convpool/main.py
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from gradio.routes import mount_gradio_app
from .sender_app import build_sender_blocks
from .receiver_app import build_receiver_blocks

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/sender")

# 외부 모듈에서 Blocks를 가져와 mount (인라인 정의 절대 금지)
sender = build_sender_blocks()
receiver = build_receiver_blocks()
mount_gradio_app(app, sender, path="/sender")
mount_gradio_app(app, receiver, path="/receiver")
