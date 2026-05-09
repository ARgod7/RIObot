// mic_bridge.js
// Browser microphone → WebSocket → Python pipeline
// Sends: transcript text + raw audio chunks
// Auto-reconnects if WebSocket drops

(function() {
  const WS_URL = "ws://localhost:8765";
  const micBtn  = document.getElementById("micBtn");
  const micDot  = document.getElementById("micDot");

  let ws         = null;
  let recognition = null;
  let mediaRecorder = null;
  let isListening = false;

  // ── WebSocket is managed by app.js ───────────────────────────────────
  // mic_bridge.js just uses the same socket via a shared getter
  // We wait for app.js to create window._rioWS, or create our own ref
  function getWS() {
    // app.js sets window._rioWS after connect
    return window._rioWS || null;
  }

  // ── Speech Recognition (transcript) ──────────────────────────────────
  function initRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.warn("[MicBridge] SpeechRecognition not supported in this browser.");
      if (micBtn) micBtn.title = "Voice not supported — use text input";
      return null;
    }

    const rec = new SpeechRecognition();
    rec.continuous      = true;
    rec.interimResults  = true;
    rec.lang            = "en-IN";   // works for both English and Hindi

    rec.onresult = (event) => {
      let finalText = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          finalText += event.results[i][0].transcript.trim();
        }
      }
      if (finalText) {
        console.log("[MicBridge] Final transcript:", finalText);
        sendTranscript(finalText);
      }
    };

    rec.onerror = (e) => {
      if (e.error !== "no-speech") {
        console.warn("[MicBridge] Recognition error:", e.error);
      }
    };

    rec.onend = () => {
      // Auto-restart if still supposed to be listening
      if (isListening) {
        setTimeout(() => { try { rec.start(); } catch(e) {} }, 300);
      }
    };

    return rec;
  }

  // ── Audio streaming (for voice emotion) ──────────────────────────────
  async function startAudioStream() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

      mediaRecorder.ondataavailable = async (e) => {
        if (e.data.size === 0) return;
        const ws = getWS();
        if (!ws || ws.readyState !== WebSocket.OPEN) return;

        // Convert blob to base64 and send
        const reader = new FileReader();
        reader.onloadend = () => {
          const base64 = reader.result.split(",")[1];
          ws.send(JSON.stringify({ type: "audio", data: base64 }));
        };
        reader.readAsDataURL(e.data);
      };

      // Send chunks every 1 second
      mediaRecorder.start(1000);
      console.log("[MicBridge] Audio stream started");
    } catch(e) {
      console.warn("[MicBridge] Mic access denied or unavailable:", e.message);
    }
  }

  function stopAudioStream() {
    if (mediaRecorder && mediaRecorder.state !== "inactive") {
      mediaRecorder.stop();
      mediaRecorder = null;
    }
  }

  // ── Send transcript via WebSocket ─────────────────────────────────────
  function sendTranscript(text) {
    const ws = getWS();
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      console.warn("[MicBridge] WS not open, cannot send transcript");
      return;
    }
    ws.send(JSON.stringify({ type: "transcript", text }));
  }

  // ── Mic button toggle ─────────────────────────────────────────────────
  function startListening() {
    isListening = true;
    if (!recognition) recognition = initRecognition();
    if (recognition) {
      try { recognition.start(); } catch(e) {}
    }
    startAudioStream();
    if (micBtn) {
      micBtn.textContent = "🔴";
      micBtn.title = "Listening... click to stop";
    }
  }

  function stopListening() {
    isListening = false;
    if (recognition) { try { recognition.stop(); } catch(e) {} }
    stopAudioStream();
    if (micBtn) {
      micBtn.textContent = "🎤";
      micBtn.title = "Click to speak";
    }
  }

  if (micBtn) {
    micBtn.addEventListener("click", () => {
      isListening ? stopListening() : startListening();
    });
  }

  // ── Expose WS to app.js ───────────────────────────────────────────────
  // app.js should set window._rioWS = ws after connecting
  // We patch connect() timing by polling
  const _origConnect = window._rioConnect;

  // Auto-start listening once page loads (optional — remove if unwanted)
  // window.addEventListener("load", () => setTimeout(startListening, 2000));

  console.log("[MicBridge] Ready. Click 🎤 to start voice input.");
})();