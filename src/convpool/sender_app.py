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
LAST_NPY = INBOX / "last_input.npy"
MANIFEST = INBOX / "manifest.json"   # {"ts": 1730000000}


def _extract_image_from_editor(img):
    """
    Gradio ImageEditor는 dict 형태로 오기도 함: {composite | image | layers}
    """
    if isinstance(img, dict):
        if img.get("composite") is not None:
            return img["composite"]
        if img.get("image") is not None:
            return img["image"]
        layers = img.get("layers")
        if isinstance(layers, (list, tuple)) and len(layers) > 0:
            return layers[-1]
        return None
    return img


def _preprocess_to_28x28(img):
    img = _extract_image_from_editor(img)
    if img is None:
        return None

    # RGB -> Gray(0~1) -> 반전 -> 이진 -> bbox -> 28x28
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
    print("[sender_app] build_sender_blocks loaded")
    with gr.Blocks(title="Sender") as app:
        gr.Markdown("## Sender\n숫자를 그리고 **Send**를 누르세요. Receiver가 자동으로 보여줍니다.")
        draw = gr.ImageEditor(type="numpy", label="Draw here", height=420, width=420)
        send_btn = gr.Button("Send", variant="primary")
        status = gr.Markdown("")
        send_btn.click(lambda: "📤 sending…", None, status, show_progress=False, queue=False)\
                .then(save_and_send, inputs=draw, outputs=status, show_progress=False, queue=False)
    return app
