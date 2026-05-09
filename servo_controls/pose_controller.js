const { SerialPort } = require('serialport');
const readline = require('readline');
const fs = require('fs');
const path = require('path');

// --- Configuration ---
const COMport = 5; 
const BAUD_RATE = 9600;
const POSES_FILE = path.join(__dirname, 'poses.json');

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
// Maps pose.json servo names to the 8-servo array indices
// Matches the order expected by the serial protocol
const servoMapping = {
    'hh': 0,      // head horizontal
    'rsv': 1,     // right shoulder vertical
    'lsv': 2,     // left shoulder vertical
    'rsh': 3,     // right shoulder horizontal
    'lsh': 4,     // left shoulder horizontal
    're': 5,      // right elbow
    'le': 6,      // left elbow
    'base': 7     // base
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
    // Create array of 8 servos, defaulting to 90 degrees
    let servoArray = new Array(8).fill(90);

    // Map each servo from the pose data
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
    
    for (let i = 0; i < servoArray.length; i++) {
        // Divide by 10 and floor
        let reducedValue = Math.floor(servoArray[i] / 10);
        
        // Pad to 2 digits (e.g. 5 -> "05")
        if (reducedValue >= 10) {
            sendData += String(reducedValue);
        } else {
            sendData += String(0) + String(reducedValue);
        }
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
