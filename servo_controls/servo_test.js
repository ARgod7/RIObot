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
// Maps user-friendly names to the index in the *sent* data array (0-7)
const servoMap = {
    'head': 0,
    'head horizontal': 0,
    'right shoulder vertical': 1,
    'rsv': 1,
    'left shoulder vertical': 2,
    'lsv': 2,
    'right shoulder horizontal': 3,
    'rsh': 3,
    'left shoulder horizontal': 4,
    'lsh': 4,
    'right elbow': 5,
    're': 5,
    'left elbow': 6,
    'le': 6,
    'base': 7
};

// --- Helper: Print Available Names ---
function printServoList() {
    console.log("\n--- Available Servo Names ---");
    
    // We reverse the map to group aliases by their ID (0-7)
    const groupedNames = {};
    for (const [name, index] of Object.entries(servoMap)) {
        if (!groupedNames[index]) groupedNames[index] = [];
        groupedNames[index].push(name);
    }

    // Print each group
    Object.keys(groupedNames).sort().forEach(id => {
        console.log(`ID ${id}: [ ${groupedNames[id].join(', ')} ]`);
    });
    console.log("-----------------------------\n");
}

// --- Main Logic ---
console.log(`--- Servo Tester Initialized on COM${COMport} ---`);

// 1. Print the names immediately
printServoList();

console.log("Enter command in format: <servo_name> <degrees>");
console.log("Examples: 'head 45', 'base 120', 'rsv 180'");
console.log("All other servos will reset to 90 degrees.");

function promptUser() {
    rl.question('Command: ', (input) => {
        handleInput(input);
        promptUser(); 
    });
}

function handleInput(input) {
    const parts = input.trim().split(/\s+/);
    
    // Extract degrees (last part of the string)
    const degreesStr = parts.pop();
    const degrees = parseInt(degreesStr);
    
    // Rejoin the rest to get the name
    const servoName = parts.join(' ').toLowerCase();

    if (isNaN(degrees)) {
        console.log("Error: Invalid degree value. Format: name degrees");
        return;
    }

    if (!servoMap.hasOwnProperty(servoName)) {
        console.log(`Error: Unknown servo name '${servoName}'. Please check the list above.`);
        return;
    }

    const targetIndex = servoMap[servoName];

    // 1. Create array of 8 servos (indices 0-7), defaulting all to 90 degrees
    let currentPose = new Array(8).fill(90);

    // 2. Set the specific servo to the requested degrees
    currentPose[targetIndex] = degrees;

    // 3. Format data according to your backend protocol
    let sendData = "";
    
    for (let i = 0; i < currentPose.length; i++) {
        // Divide by 10 and floor
        let reducedValue = Math.floor(currentPose[i] / 10);
        
        // Pad to 2 digits (e.g. 5 -> "05")
        if (reducedValue >= 10) {
            sendData += String(reducedValue);
        } else {
            sendData += String(0) + String(reducedValue);
        }
    }

    // 4. Send over Serial
    port.write(sendData, (err) => {
        if (err) {
            return console.log('Error on write: ', err.message);
        }
        console.log(`Sent: ${sendData} (Moved '${servoName}' to ${degrees}°)`);
    });
}

// Start the loop
promptUser();