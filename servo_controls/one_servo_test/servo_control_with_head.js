const { SerialPort } = require('serialport');
const readline = require('readline');

// --- Configuration ---
const COMport = 5;
const BAUD_RATE = 9600;

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

// --- Servo Mapping ---
// Maps user-friendly names to the index in the sent data array (0-8)
const servoMap = {
    'head': 0,
    'head horizontal': 0,
    'hh': 0,

    'head vertical': 1,
    'hv': 1,

    'right shoulder vertical': 2,
    'rsv': 2,

    'left shoulder vertical': 3,
    'lsv': 3,

    'right shoulder horizontal': 4,
    'rsh': 4,

    'left shoulder horizontal': 5,
    'lsh': 5,

    'right elbow': 6,
    're': 6,

    'left elbow': 7,
    'le': 7,

    'base': 8
};

// Persistent pose state in memory.
const currentPose = new Array(9).fill(90);

// --- Helper: Print Available Names ---
function printServoList() {
    console.log("\n--- Available Servo Names ---");

    const groupedNames = {};
    for (const [name, index] of Object.entries(servoMap)) {
        if (!groupedNames[index]) groupedNames[index] = [];
        groupedNames[index].push(name);
    }

    Object.keys(groupedNames).sort().forEach(id => {
        console.log(`ID ${id}: [ ${groupedNames[id].join(', ')} ]`);
    });
    console.log("-----------------------------\n");
}

function buildSendData(pose) {
    let sendData = "";

    for (let i = 0; i < pose.length; i++) {
        const reducedValue = Math.floor(pose[i] / 10);

        if (reducedValue >= 10) {
            sendData += String(reducedValue);
        } else {
            sendData += `0${reducedValue}`;
        }
    }

    return sendData;
}

// --- Main Logic ---
console.log(`--- Servo Controller Initialized on COM${COMport} ---`);
printServoList();

console.log("Enter command in format: <servo_name> <degrees>");
console.log("Examples: 'head 45', 'head vertical 30', 'base 120', 'rsv 180'");
console.log("All servos keep their previous values until changed.");

function promptUser() {
    rl.question('Command: ', (input) => {
        handleInput(input);
        promptUser();
    });
}

function handleInput(input) {
    const parts = input.trim().split(/\s+/);

    if (parts.length < 2) {
        console.log('Error: Please enter a servo name followed by degrees.');
        return;
    }

    const degreesStr = parts.pop();
    const degrees = parseInt(degreesStr);
    const servoName = parts.join(' ').toLowerCase();

    if (isNaN(degrees)) {
        console.log('Error: Invalid degree value.');
        return;
    }

    if (!servoMap.hasOwnProperty(servoName)) {
        console.log(`Error: Unknown servo name '${servoName}'.`);
        return;
    }

    const targetIndex = servoMap[servoName];
    currentPose[targetIndex] = degrees;

    const sendData = buildSendData(currentPose);

    port.write(sendData, (err) => {
        if (err) {
            return console.log('Error on write: ', err.message);
        }

        console.log(`Sent: ${sendData} (Updated '${servoName}' to ${degrees}°)`);
    });
}

promptUser();