# src/convpool/sender_app.py
from pathlib import Path
import time, json
import numpy as np
import cv2
import gradio as gr

BASE = Path(__file__).resolve().parents[2]
INBOX = BASE / "inbox"
INBOX.mkdir(parents=True, exist_ok=True)

LAST_PNG = INBOX / "last_input.png"
LAST_NPY  = INBOX / "last_input.npy"
MANIFEST  = INBOX / "manifest.json"   # {"ts": 1730000000}

def _preprocess_to_28x28(img):
    # Gradio ImageEditor는 dict로 올 수 있음: {composite | image | layers}
    if isinstance(img, dict):
        if "composite" in img and img["composite"] is not None:
            img = img["composite"]
        elif "image" in img and img["image"] is not None:
            img = img["image"]
        elif "layers" in img and isinstance(img["layers"], (list, tuple)) and len(img["layers"]) > 0:
            img = img["layers"][-1]
        else:
            img = None

    if img is None:
        return None

    # 이하 동일 (그레이스케일 → 반전 → 이진화 → 바운딩박스 → 28×28 배치)
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
    print("[sender_app] build_sender_blocks loaded")  # shows in server console
    with gr.Blocks(title="Sender") as app:
        gr.Markdown("## Sender\nDraw a digit and click **Send**. Receiver will auto-refresh.")
        draw = gr.ImageEditor(type="numpy", label="Draw here", height=420, width=420)
        send_btn = gr.Button("Send", variant="primary")
        status = gr.Markdown("")
        # instant feedback -> then save
        send_btn.click(lambda: "📤 sending…", None, status, show_progress=False, queue=False)\
                .then(save_and_send, inputs=draw, outputs=status, show_progress=False, queue=False)
    return app
