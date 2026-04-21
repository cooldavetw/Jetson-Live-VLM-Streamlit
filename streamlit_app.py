"""
Streamlit live camera demo:
- Start the webcam in the browser
- Capture the latest frame every N seconds
- Send it to an OpenAI-compatible VLM (e.g. moondream via Ollama)
- Show responses on the right as a dialogue

pip install streamlit openai pillow
"""

import base64
import io
import time
from dataclasses import dataclass
from typing import Tuple

from openai import OpenAI
from PIL import Image
import streamlit as st


CAMERA_COMPONENT = st.components.v2.component(
    "browser_camera_snapshot",
    html="""
<div class="camera-shell">
  <div class="camera-preview-wrap">
    <video class="camera-preview" autoplay playsinline muted></video>
    <div class="camera-placeholder">Camera inactive</div>
  </div>
  <canvas class="camera-canvas" aria-hidden="true"></canvas>
  <div class="camera-controls">
    <button class="camera-start" type="button">Start</button>
    <button class="camera-stop" type="button" disabled>Stop</button>
    <button class="camera-snapshot" type="button" disabled>Capture now</button>
    <select class="camera-device-select" disabled>
      <option value="">Default camera</option>
    </select>
  </div>
  <div class="camera-status">Waiting to start camera.</div>
</div>
""",
    css="""
:host {
  display: block;
}

.camera-shell {
  color: var(--st-text-color);
  font-family: var(--st-font, sans-serif);
}

.camera-preview-wrap {
  position: relative;
  width: 100%;
  aspect-ratio: 16 / 9;
  border-radius: 0.75rem;
  overflow: hidden;
  background: #0b1220;
  border: 1px solid color-mix(in srgb, var(--st-text-color) 14%, transparent);
}

.camera-preview {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: none;
  background: #000;
}

.camera-preview.is-active {
  display: block;
}

.camera-placeholder {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  color: color-mix(in srgb, white 75%, transparent);
  font-size: 1rem;
  letter-spacing: 0.02em;
}

.camera-placeholder.is-hidden {
  display: none;
}

.camera-controls {
  display: flex;
  gap: 0.75rem;
  align-items: center;
  flex-wrap: wrap;
  margin-top: 0.9rem;
}

.camera-controls button,
.camera-controls select {
  min-height: 2.5rem;
  border-radius: 0.7rem;
  border: 1px solid color-mix(in srgb, var(--st-text-color) 16%, transparent);
  background: var(--st-secondary-background-color);
  color: var(--st-text-color);
  padding: 0 0.9rem;
  font: inherit;
}

.camera-controls button {
  cursor: pointer;
}

.camera-controls button:disabled,
.camera-controls select:disabled {
  opacity: 0.55;
  cursor: not-allowed;
}

.camera-start {
  background: var(--st-primary-color);
  color: white;
  border-color: transparent;
}

.camera-status {
  margin-top: 0.75rem;
  color: color-mix(in srgb, var(--st-text-color) 75%, transparent);
  font-size: 0.95rem;
}

.camera-canvas {
  display: none;
}
""",
    js="""
const INSTANCES = new WeakMap();

function createInstance(component) {
  const { parentElement } = component;
  const root = parentElement.querySelector(".camera-shell");
  const video = root.querySelector(".camera-preview");
  const placeholder = root.querySelector(".camera-placeholder");
  const canvas = root.querySelector(".camera-canvas");
  const startBtn = root.querySelector(".camera-start");
  const stopBtn = root.querySelector(".camera-stop");
  const snapshotBtn = root.querySelector(".camera-snapshot");
  const deviceSelect = root.querySelector(".camera-device-select");
  const statusEl = root.querySelector(".camera-status");

  const instance = {
    component,
    root,
    video,
    placeholder,
    canvas,
    startBtn,
    stopBtn,
    snapshotBtn,
    deviceSelect,
    statusEl,
    stream: null,
    timerId: null,
    currentDeviceId: "",
    lastCaptureTs: 0,
    lastStatus: null,
    lastPlaying: null,
  };

  instance.setStatus = (message, isError = false) => {
    statusEl.textContent = message;
    statusEl.style.color = isError
      ? "var(--st-red-70)"
      : "color-mix(in srgb, var(--st-text-color) 75%, transparent)";
    const playing = Boolean(instance.stream);
    if (instance.lastStatus !== message) {
      instance.component.setStateValue("status", message);
      instance.lastStatus = message;
    }
    if (instance.lastPlaying !== playing) {
      instance.component.setStateValue("playing", playing);
      instance.lastPlaying = playing;
    }
  };

  instance.setPreviewActive = (active) => {
    video.classList.toggle("is-active", active);
    placeholder.classList.toggle("is-hidden", active);
    stopBtn.disabled = !active;
    snapshotBtn.disabled = !active;
  };

  instance.stopStream = () => {
    if (instance.timerId) {
      clearInterval(instance.timerId);
      instance.timerId = null;
    }
    if (instance.stream) {
      instance.stream.getTracks().forEach((track) => track.stop());
      instance.stream = null;
    }
    video.srcObject = null;
    instance.setPreviewActive(false);
    instance.setStatus("Camera stopped.");
  };

  instance.captureFrame = (source) => {
    if (!instance.stream || video.readyState < 2) {
      instance.setStatus("Camera frame is not ready yet.", true);
      return;
    }

    const width = video.videoWidth || 640;
    const height = video.videoHeight || 360;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, width, height);
    const imageDataUrl = canvas.toDataURL("image/jpeg", 0.85);
    const ts = Date.now();
    instance.lastCaptureTs = ts;
    instance.component.setTriggerValue("capture", {
      image_data_url: imageDataUrl,
      ts,
      source,
    });
    instance.setStatus(
      source === "auto"
        ? "Captured frame and sent it to the app."
        : "Captured frame."
    );
  };

  instance.scheduleCapture = () => {
    if (instance.timerId) {
      clearInterval(instance.timerId);
      instance.timerId = null;
    }
    if (!instance.stream) {
      return;
    }

    const intervalSec = Math.max(1, Number(instance.component.data?.interval_seconds || 5));
    instance.timerId = window.setInterval(() => {
      instance.captureFrame("auto");
    }, intervalSec * 1000);
  };

  instance.refreshDevices = async () => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      deviceSelect.disabled = true;
      return;
    }

    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = devices.filter((device) => device.kind === "videoinput");
      const previousValue = instance.currentDeviceId;
      deviceSelect.innerHTML = "";

      const fallbackOption = document.createElement("option");
      fallbackOption.value = "";
      fallbackOption.textContent = "Default camera";
      deviceSelect.appendChild(fallbackOption);

      videoInputs.forEach((device, index) => {
        const option = document.createElement("option");
        option.value = device.deviceId;
        option.textContent = device.label || `Camera ${index + 1}`;
        deviceSelect.appendChild(option);
      });

      deviceSelect.value = previousValue;
      deviceSelect.disabled = videoInputs.length === 0;
    } catch (error) {
      instance.setStatus(`Could not list cameras: ${error.message}`, true);
    }
  };

  instance.startStream = async (deviceId = "") => {
    instance.stopStream();
    instance.currentDeviceId = deviceId;
    instance.setStatus("Requesting camera access...");

    try {
      const constraints = {
        video: deviceId
          ? { deviceId: { exact: deviceId } }
          : true,
        audio: false,
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      instance.stream = stream;
      video.srcObject = stream;
      await video.play();
      instance.setPreviewActive(true);
      await instance.refreshDevices();
      if (instance.currentDeviceId) {
        deviceSelect.value = instance.currentDeviceId;
      }
      instance.setStatus(
        `Camera active. Capturing every ${Math.max(1, Number(instance.component.data?.interval_seconds || 5))} seconds.`
      );
      instance.scheduleCapture();
    } catch (error) {
      instance.stopStream();
      instance.setStatus(`Camera error: ${error.message}`, true);
    }
  };

  startBtn.addEventListener("click", () => {
    instance.startStream(deviceSelect.value);
  });

  stopBtn.addEventListener("click", () => {
    instance.stopStream();
  });

  snapshotBtn.addEventListener("click", () => {
    instance.captureFrame("manual");
  });

  deviceSelect.addEventListener("change", () => {
    instance.startStream(deviceSelect.value);
  });

  instance.setPreviewActive(false);
  instance.refreshDevices();
  return instance;
}

export default function(component) {
  let instance = INSTANCES.get(component.parentElement);
  if (!instance) {
    instance = createInstance(component);
    INSTANCES.set(component.parentElement, instance);
  }

  instance.component = component;
  if (instance.stream) {
    instance.scheduleCapture();
    instance.setStatus(
      `Camera active. Capturing every ${Math.max(1, Number(component.data?.interval_seconds || 5))} seconds.`
    );
  }
}
""",
)


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


def decode_image_data_url(image_data_url: str) -> Image.Image:
    header, encoded = image_data_url.split(",", 1)
    if ";base64" not in header:
        raise ValueError("Unsupported image data URL format.")
    image_bytes = base64.b64decode(encoded)
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def add_history_message(text: str, latency: float, source: str) -> None:
    st.session_state.vlm_latest = {
        "text": text,
        "latency": latency,
        "ts": time.time(),
        "source": source,
    }
    st.session_state.vlm_history.append(st.session_state.vlm_latest.copy())


def run_inference(cfg: VLMConfig, image: Image.Image, source: str) -> None:
    try:
        resized = image.resize((640, 360))
        latest_text, latest_latency = call_vlm(cfg, resized)
    except Exception as exc:
        latest_text = f"Error: {exc}"
        latest_latency = 0.0

    add_history_message(latest_text, latest_latency, source)


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
    if "vlm_history" not in st.session_state:
        st.session_state.vlm_history = []
    if "capture_interval_sec" not in st.session_state:
        st.session_state.capture_interval_sec = 5

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
        camera_result = CAMERA_COMPONENT(
            key="browser_camera",
            data={"interval_seconds": st.session_state.capture_interval_sec},
            on_capture_change=lambda: None,
            on_status_change=lambda: None,
            on_playing_change=lambda: None,
        )
        st.caption(
            f"When the webcam is active, the latest frame is sent to the VLM every "
            f"{st.session_state.capture_interval_sec} seconds."
        )
        if getattr(camera_result, "status", None):
            st.caption(camera_result.status)

    capture_event = getattr(camera_result, "capture", None)
    if capture_event and capture_event.get("image_data_url"):
        with st.spinner("Running VLM..."):
            try:
                frame_image = decode_image_data_url(capture_event["image_data_url"])
                run_inference(cfg_snapshot, frame_image, capture_event.get("source", "auto"))
            except Exception as exc:
                add_history_message(f"Error: {exc}", 0.0, "error")

    # ---- 右側：對話視窗 ----
    with col2:
        st.subheader("VLM Inference")
        if st.button("Clear history", use_container_width=True):
            st.session_state.vlm_latest = None
            st.session_state.vlm_history = []
            st.rerun()

        if st.session_state.vlm_history:
            history_container = st.container(height=520)
            for msg in st.session_state.vlm_history:
                with history_container.chat_message("user"):
                    st.markdown(cfg_snapshot.prompt)
                with history_container.chat_message("assistant"):
                    st.markdown(msg["text"])
                    st.caption(
                        f"Latency: {msg['latency']:.0f} ms | Source: {msg.get('source', 'auto')} | "
                        f"Time: {time.strftime('%H:%M:%S', time.localtime(msg['ts']))}"
                    )
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
