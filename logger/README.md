# RIO Logger & CSV Converter

Emotion logging server for the RIO robot system. Receives real-time emotion state updates via HTTP and stores them as JSON logs. Includes a utility to convert logs to CSV for analysis and visualization.

## Setup

```bash
npm install
```

## Running the Logger Server

```bash
npm start
```

The HTTP server will listen on `http://0.0.0.0:4000` (configurable via `PORT` environment variable).

### Endpoints

- `POST /emotion_update` – Receives fused emotion vectors + source vectors (face, posture, voice)
- `POST /rio_response` – Receives RIO engine responses, expression, and TTS parameters
- `POST /stimulus` – Receives raw stimulus and RIO engine state
- `GET /health` – Health check

All POST requests are logged to `logs/*.log` files in JSONL format (JSON Lines: one JSON object per line, prefixed with ISO timestamp).

## Converting Logs to CSV

Extract emotion values from logs for visualization and analysis:

```bash
# Convert emotion_update.log → emotion_update.csv (default)
npm run tocsv:emotion

# Convert rio_response.log → rio_response.csv
npm run tocsv:response

# Convert stimulus.log → stimulus.csv
npm run tocsv:stimulus

# Manual: any log file to any output
node tocsv.js logs/emotion_update.log data/emotions.csv
```

### CSV Format

Output columns:
- `timestamp` – ISO 8601 timestamp
- `joy` – Joy emotion value (0.0–1.0)
- `sadness` – Sadness emotion value (0.0–1.0)
- `fear` – Fear emotion value (0.0–1.0)
- `disgust` – Disgust emotion value (0.0–1.0)
- `anger` – Anger emotion value (0.0–1.0)
- `surprise` – Surprise emotion value (0.0–1.0)
- `label` – Stimulus label or event identifier
- `expression` – Expression intent
- `confidence` – Confidence score

### Example CSV Output

```
timestamp,joy,sadness,fear,disgust,anger,surprise,label,expression,confidence
2025-05-13T10:15:23.456Z,0.120000,0.750000,0.200000,0.000000,0.100000,0.050000,"neutral","sad",0.850000
2025-05-13T10:15:24.512Z,0.100000,0.800000,0.150000,0.050000,0.120000,0.040000,"neutral","sad",0.880000
```

### Visualization

Use the CSV file with charting tools:
- **Excel/Google Sheets**: Import CSV, create line charts
- **Python (matplotlib/plotly)**: Plot emotion trends over time
- **Observable/D3.js**: Interactive timeline visualization

Example Python:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('emotion_update.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

plt.figure(figsize=(12, 6))
for emotion in ['joy', 'sadness', 'fear', 'disgust', 'anger', 'surprise']:
    plt.plot(df['timestamp'], df[emotion], label=emotion, linewidth=1.5)

plt.xlabel('Time')
plt.ylabel('Emotion Value')
plt.legend()
plt.title('RIO Emotion Timeline')
plt.tight_layout()
plt.savefig('emotions.png', dpi=150)
plt.show()
```

## Environment Variables

- `PORT` – Logger HTTP server port (default: 4000)
- `LOGGER_URL` – URL used by `main.py` to POST logs (default: `http://127.0.0.1:4000`)
- `LOGGER_TIMEOUT` – HTTP timeout in seconds for `main.py` (default: 1.0)

Example:
```bash
PORT=5000 npm start
```

From `main.py`:
```bash
LOGGER_URL=http://0.0.0.0:4000 python main.py
```

## Log File Locations

- `logs/emotion_update.log` – All emotion updates (most frequent)
- `logs/rio_response.log` – RIO engine responses and expressions
- `logs/stimulus.log` – Raw stimuli and RIO state

## Architecture

```
main.py (Python)
  ↓ (HTTP POST)
logger.js (Express server on :4000)
  ↓ (append JSONL)
logs/*.log (raw JSON lines)
  ↓ (tocsv.js parse & convert)
*.csv (6-column emotion data for visualization)
```
