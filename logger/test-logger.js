#!/usr/bin/env node
/**
 * test-logger.js
 * Test utility to POST sample emotion data to the logger server.
 * Helps verify the logger is running and tocsv.js works correctly.
 *
 * Usage:
 *   node test-logger.js [count]
 *   node test-logger.js 10
 */

const http = require('http');

const LOGGER_URL = process.env.LOGGER_URL || 'http://127.0.0.1:4000';
const COUNT = parseInt(process.argv[2]) || 5;

/**
 * Generate random emotion vector (6 values summing to ~1.0)
 */
function randomEmotions() {
  const raw = [
    Math.random(),
    Math.random(),
    Math.random(),
    Math.random(),
    Math.random(),
    Math.random(),
  ];
  const sum = raw.reduce((a, b) => a + b, 0);
  return {
    joy: (raw[0] / sum).toFixed(6),
    sadness: (raw[1] / sum).toFixed(6),
    fear: (raw[2] / sum).toFixed(6),
    disgust: (raw[3] / sum).toFixed(6),
    anger: (raw[4] / sum).toFixed(6),
    surprise: (raw[5] / sum).toFixed(6),
  };
}

/**
 * POST payload to logger endpoint
 */
function postToLogger(endpoint, payload) {
  return new Promise((resolve, reject) => {
    const url = new URL(endpoint, LOGGER_URL);
    const data = JSON.stringify(payload);

    const options = {
      hostname: url.hostname,
      port: url.port || 80,
      path: url.pathname,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Content-Length': data.length,
      },
    };

    const req = http.request(options, (res) => {
      let body = '';
      res.on('data', (chunk) => (body += chunk));
      res.on('end', () => {
        resolve({ status: res.statusCode, body });
      });
    });

    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

/**
 * Main test loop
 */
async function main() {
  console.log(`[test-logger] Posting ${COUNT} sample records to ${LOGGER_URL}`);

  const labels = ['calm', 'happy', 'sad', 'angry', 'scared', 'surprised'];
  const expressions = ['neutral', 'joy', 'sadness', 'anger', 'fear', 'surprise'];

  for (let i = 0; i < COUNT; i++) {
    const label = labels[i % labels.length];
    const expression = expressions[i % expressions.length];
    const emotions = randomEmotions();

    // Post emotion_update
    const emotionPayload = {
      face: { emotions, confidence: Math.random() },
      posture: { emotions, confidence: Math.random() },
      voice: { emotions, confidence: Math.random() },
      fused: { emotions, confidence: 0.8 + Math.random() * 0.2 },
      raw_emotions: emotions,
      smoothed_emotions: emotions,
      stimulus_meta: {
        label,
        trust: 0.5 + Math.random() * 0.5,
        likeness: 0.5 + Math.random() * 0.5,
        timestamp: new Date().toISOString(),
      },
    };

    try {
      await postToLogger('/emotion_update', emotionPayload);
      console.log(`  [${i + 1}/${COUNT}] /emotion_update "${label}" ✓`);
    } catch (err) {
      console.error(`  [${i + 1}/${COUNT}] /emotion_update FAILED:`, err.message);
    }

    // Post rio_response
    const responsePayload = {
      response_text: `Sample response ${i + 1}`,
      expression,
      audio_url: `/audio/response_${i}.mp3`,
      rio_state: {
        emotion_vector: emotions,
        dominant_emotion: Object.entries(emotions).sort(([, a], [, b]) => b - a)[0][0],
      },
    };

    try {
      await postToLogger('/rio_response', responsePayload);
      console.log(`  [${i + 1}/${COUNT}] /rio_response "${expression}" ✓`);
    } catch (err) {
      console.error(`  [${i + 1}/${COUNT}] /rio_response FAILED:`, err.message);
    }

    await new Promise((r) => setTimeout(r, 200)); // Small delay between posts
  }

  console.log(`[test-logger] Done! Run 'npm run tocsv:emotion' to convert logs to CSV.`);
}

main().catch(console.error);
