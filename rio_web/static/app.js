// app.js — RIO frontend logic for port 8000
// Connects to Python WebSocket on ws://localhost:8765
// Handles: transcript display, RIO response, audio playback, mic status

const WS_URL = "ws://localhost:8765";
const RECONNECT_DELAY = 3000;

let ws = null;
let reconnectTimer = null;
let audioUnlocked = false;

// ── DOM refs ──────────────────────────────────────────────────────────────
const rioResponse  = document.getElementById("rioResponse");
const emotionState = document.getElementById("emotionState");
const audioPlayer  = document.getElementById("audioPlayer");
const userInput    = document.getElementById("userInput");
const sendBtn      = document.getElementById("sendBtn");
const micBtn       = document.getElementById("micBtn");
const micDot       = document.getElementById("micDot");       // green dot
const transcriptEl = document.getElementById("transcript");   // live transcript

// ── WebSocket connection ──────────────────────────────────────────────────
function connect() {
  ws = new WebSocket(WS_URL);
  window._rioWS = ws;

  ws.onopen = () => {
    console.log("[RIO] WebSocket connected");
    setMicStatus(true);
    if (reconnectTimer) { clearTimeout(reconnectTimer); reconnectTimer = null; }
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleMessage(msg);
    } catch(e) {
      console.warn("[RIO] Bad message:", event.data);
    }
  };

  ws.onclose = () => {
    console.warn("[RIO] WebSocket closed — reconnecting...");
    setMicStatus(false);
    reconnectTimer = setTimeout(connect, RECONNECT_DELAY);
  };

  ws.onerror = (e) => {
    console.error("[RIO] WebSocket error", e);
    ws.close();
  };
}

// ── Message router ────────────────────────────────────────────────────────
function handleMessage(msg) {
  switch(msg.type) {

    // Python confirms transcript received
    case "transcript_ack":
      if (transcriptEl) transcriptEl.textContent = msg.text;
      break;

    // RIO response text from dialogue agent
    case "rio_response":
      rioResponse.textContent = msg.text;
        if (msg.expression) {
          emotionState.textContent = msg.expression;
          updateFace(msg.expression);
        }
      break;

    // Play audio — Python sends path relative to /audio/
    case "play_audio":
      playAudio(msg.url);
      break;

    // Emotion vector update for display
    case "emotion_update":
      updateEmotionDisplay(msg.emotions);
      break;
  }
}

// ── Audio playback (in browser, not media player) ─────────────────────────
function playAudio(url) {
  // Add cache-buster so browser doesn't serve stale file
  audioPlayer.pause();
  audioPlayer.currentTime = 0;
  audioPlayer.src = url + "?t=" + Date.now();
  audioPlayer.preload = "auto";
  audioPlayer.muted = false;
  audioPlayer.volume = 1.0;
  audioPlayer.style.display = "block";
  audioPlayer.load();

  audioPlayer.oncanplaythrough = () => {
    console.log("[RIO] audio canplaythrough:", audioPlayer.src);
  };
  audioPlayer.onerror = () => {
    console.error("[RIO] audio element error for src:", audioPlayer.src);
    showPlayPrompt(url);
  };

  if (!audioUnlocked) {
    console.warn("[RIO] Audio locked until user interaction");
    showPlayPrompt(url);
    return;
  }
  audioPlayer.play().catch(e => {
    // Autoplay blocked — user must interact first
    console.warn("[RIO] Autoplay blocked:", e.message);
    showPlayPrompt(url);
  });
}

let playPromptShown = false;

function showPlayPrompt(url) {
  if (playPromptShown) return;
  playPromptShown = true;
  const btn = document.createElement("button");
  btn.textContent = "▶ Play RIO response";
  btn.className = "play-prompt-btn";
  btn.onclick = () => {
    playPromptShown = false;
    audioUnlocked = true;
    playAudio(url);
    btn.remove();
  };
  document.querySelector(".response-card").appendChild(btn);
}

// ── Send text input ───────────────────────────────────────────────────────
function sendText() {
  const text = userInput.value.trim();
  audioUnlocked = true;
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "transcript", text }));
  if (transcriptEl) transcriptEl.textContent = "You: " + text;
  userInput.value = "";
}

sendBtn.addEventListener("click", sendText);
userInput.addEventListener("keydown", e => {
  audioUnlocked = true;
  if (e.key === "Enter") sendText();
});
document.addEventListener("click", () => { audioUnlocked = true; }, { once: true });

// ── Mic status indicator ──────────────────────────────────────────────────
function setMicStatus(connected) {
  if (!micDot) return;
  micDot.className = connected ? "mic-dot on" : "mic-dot off";
  micDot.title = connected ? "Connected to RIO" : "Disconnected";
}

// ── Emotion display (for details panel) ──────────────────────────────────
function updateEmotionDisplay(emotions) {
  const el = document.getElementById("emotionBars");
  if (!el || !emotions) return;
  el.innerHTML = Object.entries(emotions).map(([k, v]) => `
    <div class="ebar">
      <span class="elabel">${k}</span>
      <div class="etrack"><div class="efill" style="width:${Math.round(v*100)}%"></div></div>
      <span class="eval">${v.toFixed(2)}</span>
    </div>
  `).join("");
}
const EYE_MAP = {
  joy:      "happyEyesVeryGIF.gif",
  sadness:  "sadEyesSlightGIF.gif",
  fear:     "afraidEyesSlightGIF.gif",
  anger:    "angryEyesVeryGIF.gif",
  disgust:  "disgustedEyesSlightGIF.gif",
  surprise: "surprisedEyesSlightGIF.gif",
  calm:     "neutralEyesGIF.gif",
};
const MOUTH_MAP = {
  joy:      "happy_notSpeakGIF.gif",
  sadness:  "sad_notSpeakGIF.gif",
  fear:     "angry_afraid_notSpeakGIF.gif",
  anger:    "angry_afraid_notSpeakGIF.gif",
  disgust:  "surprised_disgust_notSpeakGIF.gif",
  surprise: "surprised_disgust_notSpeakGIF.gif",
  calm:     "neutral_notSpeakGIF.gif",
};

function updateFace(expression) {
  const eyes  = document.getElementById("eyesImg");
  const mouth = document.getElementById("mouthImg");
  if (!eyes || !mouth) return;
  const e = expression.toLowerCase();
  if (EYE_MAP[e])   eyes.src  = `/assets/animatedEyes/${EYE_MAP[e]}`;
  if (MOUTH_MAP[e]) mouth.src = `/assets/animatedNoSpeeches/${MOUTH_MAP[e]}`;
}

connect();
// ── Boot ──────────────────────────────────────────────────────────────────
