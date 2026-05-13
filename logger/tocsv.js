/**
 * tocsv.js
 * Reads JSON log files and converts them to CSV format.
 * Extracts 6 emotion values (joy, sadness, fear, disgust, anger, surprise) with timestamps
 * for historical emotion visualization and linear graph plotting.
 *
 * Usage:
 *   node tocsv.js [logfile] [outputfile]
 *   node tocsv.js emotion_update.log emotions.csv
 *   node tocsv.js        (defaults to emotion_update.log → emotion_update.csv)
 */

const fs = require('fs');
const path = require('path');

const LOG_DIR = path.join(__dirname, 'logs');
const EMOTION_KEYS = ['joy', 'sadness', 'fear', 'disgust', 'anger', 'surprise'];

/**
 * Extract emotion values from a record, searching through nested structure.
 */
function extractEmotions(record) {
  const result = {};
  for (const key of EMOTION_KEYS) {
    result[key] = 0.0;
  }

  // Search in fused emotions (emotion_update records)
  if (record.fused && record.fused.emotions) {
    for (const key of EMOTION_KEYS) {
      result[key] = parseFloat(record.fused.emotions[key]) || 0.0;
    }
    return result;
  }

  // Search in raw_emotions or smoothed_emotions
  if (record.raw_emotions) {
    for (const key of EMOTION_KEYS) {
      result[key] = parseFloat(record.raw_emotions[key]) || 0.0;
    }
    return result;
  }

  if (record.smoothed_emotions) {
    for (const key of EMOTION_KEYS) {
      result[key] = parseFloat(record.smoothed_emotions[key]) || 0.0;
    }
    return result;
  }

  // Search in stimulus (for stimulus records)
  if (record.stimulus && record.stimulus.emotions) {
    for (const key of EMOTION_KEYS) {
      result[key] = parseFloat(record.stimulus.emotions[key]) || 0.0;
    }
    return result;
  }

  // Search at top level (rio_response or other structures)
  for (const key of EMOTION_KEYS) {
    if (record[key] !== undefined) {
      result[key] = parseFloat(record[key]) || 0.0;
    }
  }

  return result;
}

/**
 * Parse JSONL format: `ISO_TIMESTAMP JSON_OBJECT`
 * Returns array of { timestamp, emotions, metadata }
 */
function parseLogFile(filePath) {
  if (!fs.existsSync(filePath)) {
    console.error(`[tocsv] Log file not found: ${filePath}`);
    return [];
  }

  const content = fs.readFileSync(filePath, 'utf-8');
  const lines = content.trim().split('\n').filter(line => line.trim());
  const records = [];

  for (let i = 0; i < lines.length; i++) {
    try {
      const line = lines[i].trim();
      // Format: "ISO_TIMESTAMP JSON_OBJECT"
      const spaceIdx = line.indexOf(' ');
      if (spaceIdx === -1) continue;

      const timestamp = line.substring(0, spaceIdx);
      const jsonStr = line.substring(spaceIdx + 1);
      const data = JSON.parse(jsonStr);

      const emotions = extractEmotions(data);

      records.push({
        timestamp: new Date(timestamp),
        emotions,
        metadata: {
          label: data.stimulus_meta?.label || data.stimulus?.label || 'unknown',
          expression: data.expression || 'neutral',
          confidence: data.fused?.confidence || data.stimulus?.confidence || 0.0,
        },
      });
    } catch (err) {
      console.warn(`[tocsv] Skipping line ${i + 1}: ${err.message}`);
    }
  }

  return records;
}

/**
 * Write records to CSV with headers:
 * timestamp, joy, sadness, fear, disgust, anger, surprise, label, expression, confidence
 */
function writeCSV(filePath, records) {
  if (records.length === 0) {
    console.warn('[tocsv] No records to write.');
    return;
  }

  const headers = ['timestamp', 'joy', 'sadness', 'fear', 'disgust', 'anger', 'surprise', 'label', 'expression', 'confidence'];
  const lines = [headers.join(',')];

  for (const record of records) {
    const row = [
      record.timestamp.toISOString(),
      record.emotions.joy.toFixed(6),
      record.emotions.sadness.toFixed(6),
      record.emotions.fear.toFixed(6),
      record.emotions.disgust.toFixed(6),
      record.emotions.anger.toFixed(6),
      record.emotions.surprise.toFixed(6),
      `"${record.metadata.label}"`,
      `"${record.metadata.expression}"`,
      record.metadata.confidence.toFixed(6),
    ];
    lines.push(row.join(','));
  }

  fs.writeFileSync(filePath, lines.join('\n'), 'utf-8');
  console.log(`[tocsv] ✓ Written ${records.length} records to ${filePath}`);
}

/**
 * Main CLI handler
 */
function main() {
  const args = process.argv.slice(2);

  // Default to emotion_update.log
  let logFile = path.join(LOG_DIR, 'emotion_update.log');
  let outputFile = path.join(LOG_DIR, 'emotion_update.csv');

  if (args.length >= 1) {
    // If first arg has no path, assume it's in logs/
    const fullLogPath = path.isAbsolute(args[0]) ? args[0] : path.join(LOG_DIR, args[0]);
    logFile = fullLogPath;
    // Remove .log extension if present and add .csv
    outputFile = fullLogPath.replace(/\.log$/, '.csv') || `${fullLogPath}.csv`;
  }

  if (args.length >= 2) {
    outputFile = path.isAbsolute(args[1]) ? args[1] : path.join(LOG_DIR, args[1]);
  }

  console.log(`[tocsv] Reading from: ${logFile}`);
  const records = parseLogFile(logFile);

  if (records.length === 0) {
    console.error('[tocsv] No valid records found.');
    process.exit(1);
  }

  console.log(`[tocsv] Parsed ${records.length} records`);
  writeCSV(outputFile, records);
}

// Export for require() usage
module.exports = {
  extractEmotions,
  parseLogFile,
  writeCSV,
};

// Run CLI if invoked directly
if (require.main === module) {
  main();
}
