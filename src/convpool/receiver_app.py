# src/convpool/receiver_app.py
from pathlib import Path
import json
import gradio as gr

BASE = Path(__file__).resolve().parents[2]
INBOX = BASE / "inbox"
LAST_PNG = INBOX / "last_input.png"
MANIFEST = INBOX / "manifest.json"

_last_ts = -1

def poll_and_load():
    """새 입력이 있으면 png 경로를 반환, 없으면 그대로 유지"""
    global _last_ts
    if not MANIFEST.exists():
        return gr.update(), "대기중… (새 입력 없음)"
    try:
        ts = int(json.loads(MANIFEST.read_text()).get("ts", -1))
    except Exception:
        return gr.update(), "대기중… (manifest 읽는 중)"
    if ts == _last_ts:
        return gr.update(), "대기중…"
    if LAST_PNG.exists():
        _last_ts = ts
        return str(LAST_PNG), f"✅ 수신 ({ts})"
    return gr.update(), "대기중… (파일 준비 중)"

def build_receiver_blocks():
    with gr.Blocks(title="Receiver") as app:
        gr.Markdown("## 📺 Receiver\nSender에서 **Send**를 누르면 자동으로 이미지를 불러옵니다.")
        with gr.Row():
            img = gr.Image(label="Latest 28x28 (preprocessed)", type="filepath", height=420)
        status = gr.Markdown("대기중…")

        # 숨은 버튼 + JS setInterval로 1초마다 폴링
        poll_btn = gr.Button(visible=False, elem_id="poll_btn")
        poll_btn.click(poll_and_load, None, [img, status], show_progress=False, queue=False)

        gr.HTML("""
        <script>
          const T = 1000;
          function tick(){
            const b = document.getElementById('poll_btn');
            if(b) b.click();
          }
          window.addEventListener('load', ()=>{ tick(); setInterval(tick, T); });
        </script>
        """)
    return app
