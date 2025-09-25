# src/convpool/utils.py
from __future__ import annotations
from pathlib import Path
import tempfile
from typing import Optional, Tuple, List

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


# =========================
# 28x28 전처리
# =========================
def preprocess_to_28x28(img_bgr: np.ndarray) -> np.ndarray:
    """
    입력: BGR(또는 GRAY) 이미지
    출력: (28, 28) float32, [0,1], 배경 0 / 글자 1 (MNIST 스타일)
    """
    if img_bgr.ndim == 3:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        g = img_bgr.copy()
    g = g.astype("float32") / 255.0

    # 배경이 밝으면 반전해서 '흰 바탕+검은 글씨'가 되게
    if g.mean() > 0.5:
        g = 1.0 - g

    # 이진화 후 bbox crop
    bin_ = (g > 0.25).astype("uint8") * 255
    cnt = cv2.findNonZero(bin_)
    if cnt is not None:
        x, y, w, h = cv2.boundingRect(cnt)
        g = g[y:y + h, x:x + w]

    # 20x20으로 줄이고 28x28 중앙 배치
    digit = cv2.resize(g, (20, 20), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype="float32")
    s = (28 - 20) // 2
    canvas[s:s + 20, s:s + 20] = digit

    m = canvas.max()
    if m > 1e-6:
        canvas /= m
    return canvas


# =========================
# PIL 유틸
# =========================
def _to_rgba(img28: np.ndarray, up: int = 12) -> Image.Image:
    img = (img28 * 255).clip(0, 255).astype("uint8")
    big = cv2.resize(img, (28 * up, 28 * up), interpolation=cv2.INTER_NEAREST)
    big_rgb = cv2.cvtColor(big, cv2.COLOR_GRAY2RGB)
    pil = Image.fromarray(big_rgb)
    return pil.convert("RGBA")


def _draw_grid(pil: Image.Image, step: int, color=(200, 200, 200, 140)):
    draw = ImageDraw.Draw(pil)
    w, h = pil.size
    for x in range(0, w + 1, step):
        draw.line([(x, 0), (x, h)], fill=color, width=1)
    for y in range(0, h + 1, step):
        draw.line([(0, y), (w, y)], fill=color, width=1)


def _overlay_text(pil: Image.Image, text: str):
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    pad = 6
    w = 8 * len(text) + 12
    draw.rectangle([(pad, pad), (pad + w, 30)], fill=(0, 0, 0, 120))
    draw.text((pad + 6, 10), text, fill=(255, 255, 255, 230), font=font)


# =========================
# 애니메이션 GIF
# =========================
def build_animation_gif(
    x28: np.ndarray,
    out_size: int = 28 * 12,   # 336 px
    fps: int = 8,              # 기본 프레임 속도 (느리게)
    end_pause_ms: int = 3000,  # 👉 마지막 프레임에서 멈춰 있을 시간(ms)
) -> str:
    """
    '그려지기 → 컨볼루션 → 풀링' 순서로 보여주는 GIF 생성.
    마지막 프레임에서 end_pause_ms 만큼 정지 후 다시 루프.
    반환: 생성된 GIF 경로
    """
    assert x28.shape == (28, 28), "x28 must be (28,28)"
    up = max(1, out_size // 28)

    frames: List[Image.Image] = []
    durations: List[int] = []
    base_duration = int(1000 / max(1, fps))

    # 1) 입력이 왼쪽에서 오른쪽으로 그려지는 연출
    for col in range(28):
        mask = np.zeros_like(x28)
        mask[:, : col + 1] = 1.0
        show = x28 * mask
        f = _to_rgba(show, up)
        _overlay_text(f, "Drawn Input")
        frames.append(f)
        durations.append(base_duration)

    # 2) 간단 엣지 컨볼루션
    k = np.array([[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]], dtype="float32")
    conv = cv2.filter2D(x28, ddepth=-1, kernel=k)
    conv = (conv - conv.min()) / (conv.max() - conv.min() + 1e-8)
    for t in np.linspace(0.0, 1.0, 10):
        mix = (1 - t) * x28 + t * conv
        f = _to_rgba(mix, up)
        _overlay_text(f, "Conv 3×3 (edge-like)")
        frames.append(f)
        durations.append(base_duration)

    # 3) 2×2 MaxPool 시각화
    pooled = cv2.resize(x28, (14, 14), interpolation=cv2.INTER_AREA)
    pooled_up = cv2.resize(pooled, (28, 28), interpolation=cv2.INTER_NEAREST)
    for t in np.linspace(0.0, 1.0, 10):
        mix = (1 - t) * x28 + t * pooled_up
        f = _to_rgba(mix, up)
        _overlay_text(f, "2×2 MaxPool (visualized)")
        _draw_grid(f, step=2 * up)
        frames.append(f)
        durations.append(base_duration)

    # 마지막 프레임에 정지 시간 추가
    if durations:
        durations[-1] = max(end_pause_ms, base_duration)

    # 저장
    tmpdir = Path(tempfile.mkdtemp(prefix="convpool_anim_"))
    out_path = tmpdir / "anim.gif"
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,     # 👈 프레임별 duration 적용
        loop=0,                 # 무한 반복
        optimize=False,
        disposal=2,
    )
    return str(out_path)


# =========================
# 확률 계산 (가중치 필수)
# =========================
def get_probabilities_with_weights(x28: np.ndarray, weights_path: Path) -> np.ndarray:
    """
    필수: weights_path가 존재해야 함. 존재하지 않거나 로드 실패하면 예외 발생.
    ⚠️ 학습 때 사용한 구조에 맞춤:
       Conv2D(32,3) → MaxPool(2,2) → Conv2D(64,3) → MaxPool(2,2) → Flatten → Dense(128) → Dense(10)
       => Flatten 입력: 5×5×64 = 1600
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"weights not found: {weights_path}")

    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),                       # 5*5*64 = 1600
        layers.Dense(128, activation="relu"),
        layers.Dense(10, activation="softmax"),
    ])
    model.load_weights(str(weights_path))
    x = x28.astype("float32")[None, ..., None]
    probs = model.predict(x, verbose=0)[0]
    return probs.astype("float32")


# =========================
# 확률 막대 이미지 (NumPy RGB)
# =========================
def build_prob_bar_image(probs: np.ndarray) -> np.ndarray:
    """
    입력: probs (10,)
    출력: HxWx3 uint8 이미지 (bar chart)
    """
    W, H = 640, 360
    pad = 40
    col_w = (W - pad * 2) // 10
    bar_w = col_w - 8
    max_h = H - pad * 2 - 20

    img = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    # 제목
    title = "Class Probabilities (0-9)"
    draw.text((pad, 10), title, fill=(30, 30, 30), font=font)

    # 축
    draw.line([(pad, H - pad), (W - pad, H - pad)], fill=(120, 120, 120), width=2)
    draw.line([(pad, pad), (pad, H - pad)], fill=(120, 120, 120), width=2)

    top = int(np.argmax(probs))
    for i, p in enumerate(probs):
        x0 = pad + i * col_w
        bh = int(max_h * float(p))
        x1 = x0 + bar_w
        y1 = H - pad
        y0 = y1 - bh
        color = (70, 140, 255) if i != top else (255, 120, 60)
        draw.rectangle([x0, y0, x1, y1], fill=color)
        # 라벨
        draw.text((x0 + bar_w // 2 - 3, H - pad + 6), str(i), fill=(20, 20, 20), font=font)
        draw.text((x0, y0 - 14), f"{p*100:.1f}%", fill=(20, 20, 20), font=font)

    # 예측 표시
    pred_str = f"pred = {top}  (p={probs[top]*100:.1f}%)"
    draw.text((W - pad - 160, pad - 26), pred_str, fill=(200, 60, 40), font=font)

    return np.asarray(img, dtype=np.uint8)
