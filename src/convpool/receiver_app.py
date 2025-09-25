# src/convpool/receiver_app.py
from pathlib import Path
import json
import cv2
import numpy as np
import gradio as gr

from .utils import (
    preprocess_to_28x28,
    build_animation_gif,
    get_probabilities_with_weights,
    build_prob_bar_image,
)

# ====== 경로 세팅 ======
BASE = Path(__file__).resolve().parents[2]
INBOX = BASE / "inbox"
INBOX.mkdir(parents=True, exist_ok=True)

LAST_PNG = INBOX / "last_input.png"   # sender가 저장
MANIFEST = INBOX / "manifest.json"    # {"ts": ...}
WEIGHTS  = BASE / "data" / "mnist_cnn.weights.h5"  # 반드시 필요

_last_ts = -1  # 최근 처리한 ts

# ====== 표시 설정 ======
GIF_SIZE = 28 * 12  # 336px
GIF_FPS  = 8        # 조금 느리게


def poll_and_render():
    """새 입력이 오면: PNG→28x28→애니(GIF)+확률막대 생성"""
    global _last_ts

    if not MANIFEST.exists():
        return gr.update(), gr.update(), "대기 중… (새 입력 없음)"

    try:
        ts = int(json.loads(MANIFEST.read_text()).get("ts", -1))
    except Exception:
        return gr.update(), gr.update(), "대기 중… (manifest 파싱 실패)"

    if ts <= _last_ts:
        return gr.update(), gr.update(), f"대기 중… (최근 ts={_last_ts})"

    if not LAST_PNG.exists():
        return gr.update(), gr.update(), "대기 중… (파일 준비 중)"

    # 1) 전처리
    img = cv2.imread(str(LAST_PNG), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return gr.update(), gr.update(), "❌ 입력 PNG 읽기 실패"
    x28 = preprocess_to_28x28(img)  # (28,28) float32

    # 2) 가중치 필수 검사 + 예측
    if not WEIGHTS.exists():
        return gr.update(), gr.update(), f"❌ 가중치가 없습니다: {WEIGHTS}"

    try:
        probs = get_probabilities_with_weights(x28, WEIGHTS)
    except Exception as e:
        return gr.update(), gr.update(), f"❌ 가중치 로드/예측 실패: {e}"

    # 3) 애니메이션 GIF + 확률 막대 이미지
    gif_path = build_animation_gif(x28, out_size=GIF_SIZE, fps=GIF_FPS)
    prob_img = build_prob_bar_image(probs)

    _last_ts = ts
    status = f"✅ 새 입력(ts={ts}) · 예측={int(np.argmax(probs))} · 렌더 완료"

    # GIF는 파일경로, 막대는 numpy 이미지 반환
    return gif_path, prob_img, status


def build_receiver_blocks():
    with gr.Blocks(title="Receiver — Conv/Pool + Prediction") as app:
        gr.Markdown("## Receiver\nSender에서 보낸 입력을 자동으로 렌더합니다.")
        with gr.Row():
            # 왼쪽: GIF 파일 경로를 직접 보여줌
            anim = gr.Image(label="Animation (Conv→Pool)", type="filepath", height=720)
            # 오른쪽: 확률 막대 (numpy 이미지)
            prob = gr.Image(label="Final Probabilities", type="numpy", height=360)

        status = gr.Markdown("대기 중…")
        timer = gr.Timer(1.0)
        timer.tick(poll_and_render, outputs=[anim, prob, status])

    return app
