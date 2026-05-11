const { SerialPort } = require('serialport');
const readline = require('readline');
const fs = require('fs');
const path = require('path');

// --- Configuration ---
const COMport = 5; 
const BAUD_RATE = 9600;
const POSES_FILE = path.join(__dirname, 'poses_generated.json');

// Setup Serial Port
const port = new SerialPort({ path: `COM${COMport}`, baudRate: BAUD_RATE }, function (err) {
    if (err) {
        return console.log('Error opening port: ', err.message);
    }
});

// Setup Readline for User Input
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

// --- Load Poses ---
let poses = {};
try {
    const data = fs.readFileSync(POSES_FILE, 'utf8');
    poses = JSON.parse(data);
    console.log(`Loaded ${Object.keys(poses).length} emotions from poses.json`);
} catch (err) {
    console.error('Error loading poses.json:', err.message);
    process.exit(1);
}

// --- Servo Mapping ---
// Maps pose.json servo names to the 11-slot array indices
// Transmission layout (indices 0-10):
// 0 hh, 1 rsv, 2 lsv, 3 rsh, 4 lsh, 5 re, 6 le, 7 base, 8 hv, 9 lear, 10 rear
const servoMapping = {
    'hh': 0,
    'rsv': 1,
    'lsv': 2,
    'rsh': 3,
    'lsh': 4,
    're': 5,
    'le': 6,
    'base': 7,
    'hv': 8,
    'lear': 9,
    'rear': 10
};

// --- Helper: Print Available Emotions ---
function printAvailableEmotions() {
    console.log("\n--- Available Emotions ---");
    Object.keys(poses).forEach(emotion => {
        const sequences = Object.keys(poses[emotion]);
        console.log(`${emotion}: sequences [ ${sequences.join(', ')} ]`);
    });
    console.log("-----------------------------\n");
}

// --- Helper: Convert Pose to Servo Array ---
function poseToServoArray(poseData) {
    // Create array of 11 servos, defaulting to 90 degrees
    let servoArray = new Array(11).fill(90);

    // Map each servo from the pose data (if present)
    for (const [servoName, value] of Object.entries(poseData)) {
        if (servoMapping.hasOwnProperty(servoName)) {
            const index = servoMapping[servoName];
            servoArray[index] = value;
        }
    }

    return servoArray;
}

// --- Helper: Format and Send Data ---
function sendPose(servoArray) {
    let sendData = "";

    // Expecting 11 values -> 22 characters total
    for (let i = 0; i < servoArray.length; i++) {
        let reducedValue = Math.floor(servoArray[i] / 10);
        sendData += reducedValue >= 10 ? String(reducedValue) : `0${reducedValue}`;
    }

    // Send over Serial
    port.write(sendData, (err) => {
        if (err) {
            return console.log('Error on write: ', err.message);
        }
        console.log(`Sent: ${sendData}`);
    });
}

// --- Main Logic ---
console.log(`--- Pose Controller Initialized on COM${COMport} ---`);
printAvailableEmotions();

console.log("Enter command in format: <emotion_name> <sequence_number>");
console.log("Examples: 'joy 0', 'sadness 0', 'anger 0'\n");

function promptUser() {
    rl.question('Command: ', (input) => {
        handleInput(input);
        promptUser(); 
    });
}

function handleInput(input) {
    const parts = input.trim().split(/\s+/);
    
    if (parts.length < 2) {
        console.log("Error: Please provide emotion name and sequence number");
        return;
    }

    const emotionName = parts[0].toLowerCase();
    const sequenceStr = parts[1];
    const sequence = sequenceStr.toString();

    // Validate emotion exists
    if (!poses.hasOwnProperty(emotionName)) {
        console.log(`Error: Unknown emotion '${emotionName}'. Check the list above.`);
        return;
    }

    // Validate sequence exists
    if (!poses[emotionName].hasOwnProperty(sequence)) {
        const availableSeq = Object.keys(poses[emotionName]).join(', ');
        console.log(`Error: Sequence '${sequence}' not found for '${emotionName}'. Available: [ ${availableSeq} ]`);
        return;
    }

    // Get the pose data
    const poseData = poses[emotionName][sequence];

    // Convert to servo array
    const servoArray = poseToServoArray(poseData);

    // Send it
    console.log(`Executing: ${emotionName} (sequence ${sequence})`);
    console.log(`Servo values: [${servoArray.join(', ')}]`);
    sendPose(servoArray);
}

// Start the loop
promptUser();
