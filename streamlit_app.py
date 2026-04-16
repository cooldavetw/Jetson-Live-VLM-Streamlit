"""
Streamlit live camera demo:
- Start the webcam in the browser
- Capture the latest frame every N seconds
- Send it to an OpenAI-compatible VLM (e.g. moondream via Ollama)
- Show responses on the right as a dialogue

pip install streamlit openai pillow streamlit-autorefresh streamlit-webrtc
"""

import base64
import io
import queue
import time
from dataclasses import dataclass
from typing import Tuple

from openai import OpenAI
from PIL import Image
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import WebRtcMode, webrtc_streamer


# ---------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------
@dataclass
class VLMConfig:
    api_base: str
    api_key: str
    model: str
    prompt: str
    max_tokens: int


# ---------------------------------------------------------------------
# VLM helper
# ---------------------------------------------------------------------
def encode_image(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def call_vlm(cfg: VLMConfig, image: Image.Image) -> Tuple[str, float]:
    """
    Call an OpenAI-compatible vision endpoint and return (text, latency_ms).
    """
    client = OpenAI(base_url=cfg.api_base, api_key=cfg.api_key or "EMPTY")

    img_b64 = encode_image(image)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": cfg.prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
            ],
        }
    ]

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=cfg.model,
        messages=messages,
        max_tokens=cfg.max_tokens,
        temperature=0.7,
    )
    latency_ms = (time.perf_counter() - start) * 1000.0
    text = response.choices[0].message.content.strip()

    return text, latency_ms


def run_inference(cfg: VLMConfig, image: Image.Image) -> None:
    try:
        resized = image.resize((640, 360))
        latest_text, latest_latency = call_vlm(cfg, resized)
    except Exception as exc:
        latest_text = f"Error: {exc}"
        latest_latency = 0.0

    st.session_state.vlm_latest = {
        "text": latest_text,
        "latency": latest_latency,
        "ts": time.time(),
    }


def get_latest_camera_frame(ctx) -> Image.Image | None:
    if not ctx or not ctx.state.playing or not ctx.video_receiver:
        return None

    try:
        frame = ctx.video_receiver.get_frame(timeout=1)
    except queue.Empty:
        return None

    frame_rgb = frame.to_ndarray(format="rgb24")
    return Image.fromarray(frame_rgb)


# ---------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Streamlit VLM Snapshot Demo", layout="wide")
    st.title("Streamlit Camera + OpenAI-compatible VLM")
    st.caption("Experimental demo mirroring Live VLM WebUI (live camera → VLM → dialogue).")

    # 初始化 session_state
    if "vlm_config_snapshot" not in st.session_state:
        st.session_state.vlm_config_snapshot = VLMConfig(
            api_base="http://192.168.66.26:40960/v1",
            api_key="",
            model="vlm",
            prompt="Describe the scene in details in traditional chinese.",
            max_tokens=300,
        )

    # 最新結果 (右側顯示用)
    if "vlm_latest" not in st.session_state:
        st.session_state.vlm_latest = None  # dict: {text, latency, ts}
    if "capture_interval_sec" not in st.session_state:
        st.session_state.capture_interval_sec = 5
    if "last_auto_capture_ts" not in st.session_state:
        st.session_state.last_auto_capture_ts = 0.0

    # ---- Sidebar: config UI ----
    with st.sidebar:
        st.subheader("VLM Settings")

        cfg0: VLMConfig = st.session_state.vlm_config_snapshot

        api_base = st.text_input("API Base", cfg0.api_base)
        api_key = st.text_input("API Key (if needed)", cfg0.api_key, type="password")
        model = st.text_input("Model", cfg0.model)
        prompt = st.text_area("Prompt", cfg0.prompt, height=80)
        max_tokens = st.slider("Max tokens", 32, 1024, cfg0.max_tokens, step=32)
        capture_interval_sec = st.number_input(
            "Capture interval (seconds)",
            min_value=1,
            value=st.session_state.capture_interval_sec,
            step=1,
            help="Capture the latest webcam frame and send it to the VLM on this interval.",
        )
        st.session_state.capture_interval_sec = int(capture_interval_sec)

        # 更新 snapshot
        st.session_state.vlm_config_snapshot = VLMConfig(
            api_base=api_base,
            api_key=api_key,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
        )

    cfg_snapshot: VLMConfig = st.session_state.vlm_config_snapshot

    # 兩欄：左邊影像，右邊對話
    col1, col2 = st.columns([2, 1])

    # ---- 左側：Camera & snapshot ----
    with col1:
        st.subheader("Webcam")
        ctx = webrtc_streamer(
            key="live_camera",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )
        manual_run_btn = st.button(
            "Run VLM now",
            type="primary",
            use_container_width=True,
            disabled=not (ctx and ctx.state.playing),
        )
        st.caption(
            f"When the webcam is active, the latest frame is sent to the VLM every "
            f"{st.session_state.capture_interval_sec} seconds."
        )

    if ctx and ctx.state.playing:
        st_autorefresh(
            interval=st.session_state.capture_interval_sec * 1000,
            key="vlm_auto_capture_refresh",
        )

    # ---- Run inference when button pressed ----
    if manual_run_btn:
        with st.spinner("Running VLM..."):
            frame_image = get_latest_camera_frame(ctx)
            if frame_image is not None:
                run_inference(cfg_snapshot, frame_image)
                st.session_state.last_auto_capture_ts = time.time()
            else:
                st.session_state.vlm_latest = {
                    "text": "Error: no webcam frame available yet.",
                    "latency": 0.0,
                    "ts": time.time(),
                }
    elif ctx and ctx.state.playing:
        now = time.time()
        elapsed = now - st.session_state.last_auto_capture_ts
        if elapsed >= st.session_state.capture_interval_sec:
            frame_image = get_latest_camera_frame(ctx)
            if frame_image is not None:
                run_inference(cfg_snapshot, frame_image)
                st.session_state.last_auto_capture_ts = now
    elif st.session_state.vlm_latest:
        latest_text = st.session_state.vlm_latest["text"]
        latest_latency = st.session_state.vlm_latest["latency"]

    # ---- 右側：對話視窗 ----
    with col2:
        st.subheader("VLM Inference")

        if st.session_state.vlm_latest:
            msg = st.session_state.vlm_latest
            with st.chat_message("assistant"):
                st.markdown(msg["text"])
                st.caption(f"Latency: {msg['latency']:.0f} ms")
        else:
            st.info("Waiting for the webcam stream and first response from the VLM...")

    st.markdown(
        "Tip: run a local Ollama/vLLM/NIM endpoint with a vision model "
        "(e.g., `moondream` on Ollama) and set API Base to its `/v1` URL. "
        "Start the webcam, then the app will capture and send the latest frame to the VLM "
        "every configured interval."
    )


if __name__ == "__main__":
    main()
