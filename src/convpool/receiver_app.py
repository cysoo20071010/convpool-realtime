# src/convpool/receiver_app.py
from pathlib import Path
import json, cv2, numpy as np
import gradio as gr

# ===== PATH CONSTANTS (전역) =====
BASE = Path(__file__).resolve().parents[2]
INBOX = BASE / "inbox"
INBOX.mkdir(parents=True, exist_ok=True)
LAST_PNG = INBOX / "last_input.png"
MANIFEST = INBOX / "manifest.json"

_last_ts = -1

def poll_and_load():
    global _last_ts
    if not MANIFEST.exists():
        return gr.update(), "waiting… (no input yet)"
    try:
        ts = int(json.loads(MANIFEST.read_text()).get("ts", -1))
    except Exception:
        return gr.update(), "waiting… (manifest unreadable)"
    if ts == _last_ts:
        return gr.update(), "waiting…"
    if LAST_PNG.exists():
        _last_ts = ts
        # --- 화면용 업스케일 (정사각형 픽셀, nearest) ---
        img = cv2.imread(str(LAST_PNG), cv2.IMREAD_GRAYSCALE)
        scale = 12                        # 28*12 = 336px (원하면 숫자 수정)
        big  = cv2.resize(img, (28*scale, 28*scale), interpolation=cv2.INTER_NEAREST)
        big  = cv2.cvtColor(big, cv2.COLOR_GRAY2BGR)
        return big, f"✅ received (ts={ts})"   # ← type="numpy"로 반환
    return gr.update(), "waiting… (file not ready)"

def build_receiver_blocks():
    with gr.Blocks(title="Receiver") as app:
        gr.Markdown("## Receiver\nAuto-loads the latest sent image every second.")
        with gr.Row():
            # type을 filepath → numpy 로 변경 (위에서 numpy 반환하니까)
            img = gr.Image(label="latest (scaled view)", type="numpy", height=420)
        status = gr.Markdown("waiting…")
        timer = gr.Timer(1.0)
        timer.tick(poll_and_load, outputs=[img, status])
    return app
