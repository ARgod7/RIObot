// app.js — RIO frontend logic for port 8000
// Connects to Python WebSocket on ws://<current-host>:8765
// Handles: conversation log, RIO response, audio + speaking mouth, mic status

const WS_URL = `ws://${window.location.hostname}:8765`;
const RECONNECT_DELAY = 3000;

let ws = null;
let reconnectTimer = null;
let audioUnlocked = false;
/** If autoplay was blocked, play this URL after the next user gesture. */
let pendingAudioUrl = null;

function flushPendingAudio() {
  const url = pendingAudioUrl;
  if (!url || !audioUnlocked) return;
  pendingAudioUrl = null;
  playAudio(url);
}

/** Last dialogue expression — drives idle vs speaking mouth art */
let currentExpression = "calm";
let rioSpeaking = false;
let speakingFallbackTimer = null;

const emotionState = document.getElementById("emotionState");
const audioPlayer  = document.getElementById("audioPlayer");
const userInput    = document.getElementById("userInput");
const sendBtn      = document.getElementById("sendBtn");
const micDot       = document.getElementById("micDot");
const conversationLog = document.getElementById("conversationLog");

function hideLogPlaceholder() {
  const ph = document.getElementById("logPlaceholder");
  if (ph) ph.hidden = true;
}

function appendBubble(role, text) {
  const log = conversationLog;
  if (!log || !text) return;
  hideLogPlaceholder();
  const row = document.createElement("div");
  row.className = `msg msg-${role}`;
  const stack = document.createElement("div");
  stack.className = "msg-stack";
  const label = document.createElement("span");
  label.className = "msg-label";
  label.textContent = role === "user" ? "YOU" : "RIO";
  const bubble = document.createElement("div");
  bubble.className = "msg-bubble";
  bubble.textContent = text;
  stack.appendChild(label);
  stack.appendChild(bubble);
  row.appendChild(stack);
  log.appendChild(row);
  log.scrollTop = log.scrollHeight;
}

function sendAudioState(playing) {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "audio_state", playing: !!playing }));
}

function setRioSpeaking(on) {
  rioSpeaking = !!on;
  applyMouthState();
}

function clearSpeakingFallback() {
  if (speakingFallbackTimer) {
    clearTimeout(speakingFallbackTimer);
    speakingFallbackTimer = null;
  }
}

function armSpeakingFallback(ms = 12000) {
  clearSpeakingFallback();
  speakingFallbackTimer = setTimeout(() => {
    setRioSpeaking(false);
  }, ms);
}

audioPlayer.addEventListener("playing", () => {
  clearSpeakingFallback();
  setRioSpeaking(true);
  sendAudioState(true);
});
audioPlayer.addEventListener("ended", () => {
  clearSpeakingFallback();
  setRioSpeaking(false);
  sendAudioState(false);
});
audioPlayer.addEventListener("error", () => {
  clearSpeakingFallback();
  setRioSpeaking(false);
  sendAudioState(false);
});
audioPlayer.addEventListener("pause", () => {
  if (audioPlayer.ended) {
    clearSpeakingFallback();
    setRioSpeaking(false);
    sendAudioState(false);
  }
});

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

function handleMessage(msg) {
  switch(msg.type) {

    case "transcript_ack": {
      const raw = (msg.text || "").replace(/^You:\s*/i, "").trim();
      if (raw) appendBubble("user", raw);
      break;
    }

    case "rio_response": {
      const text = (msg.text || "").trim();
      if (text) appendBubble("rio", text);
      if (msg.expression && emotionState) {
        emotionState.textContent = msg.expression;
        updateFace(msg.expression);
      }
      break;
    }

    case "play_audio":
      playAudio(msg.url);
      break;

    case "emotion_update":
      updateEmotionDisplay(msg.emotions);
      break;
  }
}

function playAudio(url) {
  pendingAudioUrl = null;
  audioPlayer.pause();
  audioPlayer.currentTime = 0;
  audioPlayer.src = url + "?t=" + Date.now();
  audioPlayer.preload = "auto";
  audioPlayer.muted = false;
  audioPlayer.volume = 1.0;
  audioPlayer.load();

  // Must run before sendAudioState(true): a blocked client must not tell the server
  // we started/stopped playback, or stop_alive_animation() fires and desyncs other
  // browsers + the physical servo when two clients share one WS backend.
  if (!audioUnlocked) {
    console.warn("[RIO] Audio locked until user interaction — will play after a tap");
    setRioSpeaking(false);
    pendingAudioUrl = url;
    return;
  }

  setRioSpeaking(true);
  sendAudioState(true);
  armSpeakingFallback();

  audioPlayer.oncanplaythrough = () => {
    console.log("[RIO] audio canplaythrough:", audioPlayer.src);
  };
  audioPlayer.onerror = () => {
    console.error("[RIO] audio element error for src:", audioPlayer.src);
    clearSpeakingFallback();
    setRioSpeaking(false);
    sendAudioState(false);
    pendingAudioUrl = url;
  };

  audioPlayer.play().catch(e => {
    console.warn("[RIO] Autoplay blocked:", e.message);
    clearSpeakingFallback();
    setRioSpeaking(false);
    sendAudioState(false);
    pendingAudioUrl = url;
  });
}

function sendText() {
  if (!userInput) return;
  const text = userInput.value.trim();
  audioUnlocked = true;
  flushPendingAudio();
  if (!text || !ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type: "transcript", text }));
  userInput.value = "";
}

if (sendBtn) sendBtn.addEventListener("click", sendText);
if (userInput) {
  userInput.addEventListener("keydown", e => {
    audioUnlocked = true;
    flushPendingAudio();
    if (e.key === "Enter") sendText();
  });
}
function noteUserActivation() {
  audioUnlocked = true;
  flushPendingAudio();
}
document.addEventListener("click", noteUserActivation, true);
document.addEventListener("pointerdown", noteUserActivation, { capture: true, passive: true });
document.addEventListener("touchstart", noteUserActivation, { capture: true, passive: true });

/** Mic bridge calls this on 🎤 tap so mobile gets a gesture before async getUserMedia. */
window.__rioNoteUserActivation = noteUserActivation;

function setMicStatus(connected) {
  if (!micDot) return;
  micDot.className = connected ? "mic-dot on" : "mic-dot off";
  micDot.title = connected ? "Connected to RIO" : "Disconnected";
}

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

/** Speaking-loop GIFs under animatedSpeeches/ */
const SPEAKING_MAP = {
  joy:      "happySpeakingRAW.gif",
  sadness:  "sadSpeakingRAW.gif",
  fear:     "afraid_angrySpeakingRAW.gif",
  anger:    "afraid_angrySpeakingRAW.gif",
  disgust:  "surprised_disgustSpeakingRAW.gif",
  surprise: "surprised_disgustSpeakingRAW.gif",
  calm:     "neutralSpeakingRAW.gif",
};

function normalizeExpression(expr) {
  const e = String(expr || "calm").toLowerCase();
  if (EYE_MAP[e]) return e;
  return "calm";
}

function applyMouthState() {
  const mouth = document.getElementById("mouthImg");
  if (!mouth) return;
  const ex = normalizeExpression(currentExpression);
  if (rioSpeaking) {
    const file = SPEAKING_MAP[ex] || SPEAKING_MAP.calm;
    mouth.src = `/assets/animatedSpeeches/${file}`;
  } else {
    const file = MOUTH_MAP[ex] || MOUTH_MAP.calm;
    mouth.src = `/assets/animatedNoSpeeches/${file}`;
  }
}

function updateFace(expression) {
  const eyes = document.getElementById("eyesImg");
  const mouth = document.getElementById("mouthImg");
  if (!eyes || !mouth) return;
  currentExpression = normalizeExpression(expression);
  const ex = currentExpression;
  eyes.src = `/assets/animatedEyes/${EYE_MAP[ex]}`;
  applyMouthState();
}

connect();
updateFace("calm");
