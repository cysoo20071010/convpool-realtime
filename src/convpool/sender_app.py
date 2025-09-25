# src/convpool/sender_app.py
from pathlib import Path
import time, json
import numpy as np
import cv2
import gradio as gr

# 프로젝트 루트 기준 공유 폴더
BASE = Path(__file__).resolve().parents[2]
INBOX = BASE / "inbox"
INBOX.mkdir(parents=True, exist_ok=True)

LAST_PNG = INBOX / "last_input.png"
LAST_NPY  = INBOX / "last_input.npy"
MANIFEST  = INBOX / "manifest.json"   # {"ts": 1730000000}

def _preprocess_to_28x28(img):
    """Gradio ImageEditor 입력을 28x28(중앙정렬)로 정규화"""
    if isinstance(img, dict):
        img = img.get("composite") or img.get("image") or (img.get("layers")[-1] if img.get("layers") else None)
    if img is None:
        return None
    g = (img[..., :3].mean(axis=-1) / 255.0).astype("float32")
    g = 1.0 - g
    g = (g > 0.5).astype("float32")
    cnt = cv2.findNonZero((g * 255).astype("uint8"))
    if cnt is not None:
        x, y, w, h = cv2.boundingRect(cnt)
        g = g[y:y + h, x:x + w]
    digit = cv2.resize(g, (20, 20), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype="float32")
    s = (28 - 20) // 2
    canvas[s:s + 20, s:s + 20] = digit
    return canvas

def save_and_send(img):
    x28 = _preprocess_to_28x28(img)
    if x28 is None:
        return "❌ 이미지가 없습니다."
    np.save(LAST_NPY, x28)
    cv2.imwrite(str(LAST_PNG), (x28 * 255).astype("uint8"))
    MANIFEST.write_text(json.dumps({"ts": int(time.time())}))
    return "✅ 전송됨"

def build_sender_blocks():
    with gr.Blocks(title="Sender") as app:
        gr.Markdown("## 📨 Sender\n그림을 그리고 **Send**를 누르면 `/receiver`가 자동 갱신됩니다.")
        draw = gr.ImageEditor(type="numpy", label="여기에 숫자를 그리세요", height=420, width=420)
        send_btn = gr.Button("Send", variant="primary")
        status = gr.Markdown("")

        # UX: 누르는 즉시 '전송 중…' → 처리 결과
        send_btn.click(lambda: "📤 전송 중…", None, status, show_progress=False, queue=False) \
                .then(save_and_send, inputs=draw, outputs=status, show_progress=False, queue=False)
    return app
