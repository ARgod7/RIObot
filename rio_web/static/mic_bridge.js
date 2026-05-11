// mic_bridge.js
// Browser microphone → WebSocket → Python pipeline
// Sends: transcript text + raw audio chunks
// Auto-reconnects if WebSocket drops

(function() {
  const WS_URL = "ws://localhost:8765";
  const micBtn  = document.getElementById("micBtn");
  const micDot  = document.getElementById("micDot");
  const micHintEl = document.getElementById("micHint");

  function setMicBtnListening(on) {
    if (!micBtn) return;
    const glyph = micBtn.querySelector(".mic-glyph");
    const label = micBtn.querySelector(".mic-on-label");
    if (glyph) glyph.textContent = on ? "🔴" : "🎤";
    if (label) {
      if (on) {
        label.removeAttribute("hidden");
        label.textContent = "On";
      } else {
        label.setAttribute("hidden", "");
      }
    }
  }

  let ws         = null;
  let recognition = null;
  let mediaRecorder = null;
  let isListening = false;
  let audioContext = null;
  let audioProcessor = null;
  let audioSource = null;
  let lastSentTranscript = "";
  let lastInterimTranscript = "";
  let interimTimer = null;
  let lastSpeechAt = 0;
  let noSpeechTimer = null;

  function setMicHint(text) {
    if (!micHintEl) return;
    micHintEl.textContent = text;
  }

  // ── WebSocket is managed by app.js ───────────────────────────────────
  // mic_bridge.js just uses the same socket via a shared getter
  // We wait for app.js to create window._rioWS, or create our own ref
  function getWS() {
    // app.js sets window._rioWS after connect
    if (window._rioWS && window._rioWS.readyState === WebSocket.OPEN) {
      return window._rioWS;
    }

    if (ws && ws.readyState === WebSocket.OPEN) {
      return ws;
    }

    // Fallback: create our own socket if app.js isn't ready
    if (!ws || ws.readyState === WebSocket.CLOSED) {
      ws = new WebSocket(WS_URL);
      ws.onopen = () => {
        console.log("[MicBridge] Fallback WS connected");
        if (micDot) micDot.className = "mic-dot on";
      };
      ws.onclose = () => {
        console.warn("[MicBridge] Fallback WS closed");
        if (micDot) micDot.className = "mic-dot off";
      };
      ws.onerror = (e) => {
        console.warn("[MicBridge] Fallback WS error", e);
      };
    }
    return ws;
  }

  // ── Speech Recognition (transcript) ──────────────────────────────────
  function initRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.warn("[MicBridge] SpeechRecognition not supported in this browser.");
      if (micBtn) micBtn.title = "Voice not supported — use text input";
      setMicHint("Voice recognition not supported in this browser. Use Chrome/Edge or type in the input box.");
      return null;
    }

    const rec = new SpeechRecognition();
    rec.continuous      = true;
    rec.interimResults  = true;
    rec.lang            = "en-IN";   // works for both English and Hindi

    rec.onresult = (event) => {
      lastSpeechAt = Date.now();
      let finalText = "";
      let interimText = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) {
          finalText += " " + event.results[i][0].transcript.trim();
        } else {
          interimText += " " + event.results[i][0].transcript.trim();
        }
      }

      finalText = finalText.trim();
      interimText = interimText.trim();

      if (finalText) {
        console.log("[MicBridge] Final transcript:", finalText);
        sendTranscript(finalText);
        setMicHint("Sent — waiting for RIO…");
        lastInterimTranscript = "";
        if (interimTimer) {
          clearTimeout(interimTimer);
          interimTimer = null;
        }
        return;
      }

      // Fallback: some browsers rarely emit `isFinal`.
      // Send the latest interim phrase after a short pause.
      if (interimText) {
        setMicHint("Listening... " + interimText);
        lastInterimTranscript = interimText;
        if (interimTimer) clearTimeout(interimTimer);
        interimTimer = setTimeout(() => {
          if (isListening && lastInterimTranscript) {
            console.log("[MicBridge] Interim transcript fallback:", lastInterimTranscript);
            sendTranscript(lastInterimTranscript);
          }
        }, 1200);
      }
    };

    rec.onerror = (e) => {
      if (e.error !== "no-speech") {
        console.warn("[MicBridge] Recognition error:", e.error);
      }
      if (e.error === "not-allowed" || e.error === "service-not-allowed") {
        setMicHint("Microphone permission blocked. Allow mic access in browser site settings.");
      } else if (e.error === "audio-capture") {
        setMicHint("No microphone detected by browser.");
      } else if (e.error === "network") {
        setMicHint("Speech recognition network error. Check internet and retry.");
      } else if (e.error !== "no-speech") {
        setMicHint("Recognition error: " + e.error);
      }
    };

    rec.onstart = () => {
      lastSpeechAt = Date.now();
      setMicHint("Mic is on. Speak now...");
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
      audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      audioSource = audioContext.createMediaStreamSource(stream);
      audioProcessor = audioContext.createScriptProcessor(4096, 1, 1);

      audioProcessor.onaudioprocess = (e) => {
        const ws = getWS();
        if (!ws || ws.readyState !== WebSocket.OPEN) return;

        const input = e.inputBuffer.getChannelData(0);
        const pcm16 = new Int16Array(input.length);
        for (let i = 0; i < input.length; i++) {
          const s = Math.max(-1, Math.min(1, input[i]));
          pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
        }

        const bytes = new Uint8Array(pcm16.buffer);
        let binary = "";
        for (let i = 0; i < bytes.length; i++) {
          binary += String.fromCharCode(bytes[i]);
        }
        const base64 = btoa(binary);
        ws.send(JSON.stringify({ type: "audio", data: base64, sampleRate: 16000 }));
      };

      audioSource.connect(audioProcessor);
      audioProcessor.connect(audioContext.destination);
      console.log("[MicBridge] Audio stream started (PCM16)");
    } catch(e) {
      console.warn("[MicBridge] Mic access denied or unavailable:", e.message);
    }
  }

  function stopAudioStream() {
    if (audioProcessor) {
      audioProcessor.disconnect();
      audioProcessor = null;
    }
    if (audioSource) {
      audioSource.disconnect();
      audioSource = null;
    }
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }
  }

  // ── Send transcript via WebSocket ─────────────────────────────────────
  function sendTranscript(text) {
    const normalized = String(text || "").trim();
    if (!normalized) return;
    // Ignore immediate duplicate sends from final/interim overlap.
    if (normalized.toLowerCase() === lastSentTranscript.toLowerCase()) {
      return;
    }

    const ws = getWS();
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      console.warn("[MicBridge] WS not open, cannot send transcript");
      return;
    }
    ws.send(JSON.stringify({ type: "transcript", text: normalized }));
    lastSentTranscript = normalized;
  }

  // ── Mic button toggle ─────────────────────────────────────────────────
  function startListening() {
    isListening = true;
    if (typeof window.openRioChat === "function") window.openRioChat();
    lastSpeechAt = Date.now();
    setMicHint("Starting microphone...");
    if (!recognition) recognition = initRecognition();
    if (recognition) {
      try {
        recognition.start();
      } catch(e) {
        // If engine is in a bad state from prior session, reset and retry once.
        try { recognition.stop(); } catch(_) {}
        setTimeout(() => {
          if (!isListening || !recognition) return;
          try { recognition.start(); } catch(_) {}
        }, 250);
      }
    }
    startAudioStream();
    if (noSpeechTimer) clearInterval(noSpeechTimer);
    noSpeechTimer = setInterval(() => {
      if (!isListening) return;
      if (Date.now() - lastSpeechAt > 8000) {
        setMicHint("No speech detected yet. Check mic permission and speak closer to mic.");
      }
    }, 1000);
    setMicBtnListening(true);
    if (micBtn) micBtn.title = "Listening... click to stop";
  }

  function stopListening() {
    isListening = false;
    // Flush any pending interim words before shutting down.
    if (lastInterimTranscript) {
      sendTranscript(lastInterimTranscript);
    }
    if (recognition) { try { recognition.stop(); } catch(e) {} }
    if (interimTimer) {
      clearTimeout(interimTimer);
      interimTimer = null;
    }
    if (noSpeechTimer) {
      clearInterval(noSpeechTimer);
      noSpeechTimer = null;
    }
    lastInterimTranscript = "";
    stopAudioStream();
    setMicHint("Mic stopped.");
    setMicBtnListening(false);
    if (micBtn) micBtn.title = "Click to speak";
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