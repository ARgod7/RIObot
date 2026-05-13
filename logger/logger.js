const express = require('express');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(express.json({ limit: '2mb' }));

const LOG_DIR = path.join(__dirname, 'logs');
if (!fs.existsSync(LOG_DIR)) fs.mkdirSync(LOG_DIR, { recursive: true });

function writeLog(name, data) {
  const file = path.join(LOG_DIR, `${name}.log`);
  const line = `${new Date().toISOString()} ${JSON.stringify(data)}\n`;
  fs.appendFile(file, line, (err) => {
    if (err) console.error('[logger] Write failed', err);
  });
}

app.post('/emotion_update', (req, res) => {
  const payload = req.body || {};
  console.log('[logger] /emotion_update', payload.stimulus_meta?.label || '');
  writeLog('emotion_update', payload);
  res.json({ ok: true });
});

app.post('/rio_response', (req, res) => {
  const payload = req.body || {};
  console.log('[logger] /rio_response', payload.expression || '');
  writeLog('rio_response', payload);
  res.json({ ok: true });
});

app.post('/stimulus', (req, res) => {
  const payload = req.body || {};
  console.log('[logger] /stimulus', payload.stimulus?.label || '');
  writeLog('stimulus', payload);
  res.json({ ok: true });
});

app.get('/health', (req, res) => res.json({ ok: true }));

const PORT = process.env.PORT || 4000;
app.listen(PORT, () => console.log(`[logger] Listening on http://0.0.0.0:${PORT}`));

module.exports = app;
