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

// --- Servo Mapping (RAW INDICES 0-8) ---
// We map these to their "true" positions before filtering.
// Index 1 (Head Vertical) is included here but will be skipped during transmission.
const servoMap = {
    'head': 0,
    'head horizontal': 0,
    'hh': 0,
    
    'head vertical': 1, // This will be EXCLUDED in transmission
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

// --- Helper: Print Available Names ---
function printServoList() {
    console.log("\n--- Available Servo Names ---");
    const groupedNames = {};
    for (const [name, index] of Object.entries(servoMap)) {
        if (!groupedNames[index]) groupedNames[index] = [];
        groupedNames[index].push(name);
    }
    Object.keys(groupedNames).sort().forEach(id => {
        let note = (id == 1) ? " (Will be SKIPPED)" : "";
        console.log(`ID ${id}${note}: [ ${groupedNames[id].join(', ')} ]`);
    });
    console.log("-----------------------------\n");
}

// --- Main Logic ---
console.log(`--- Servo Tester Initialized on COM${COMport} ---`);
printServoList();
console.log("Enter command in format: <servo_name> <degrees>");
console.log("Examples: 'head 45', 'rsv 180'");
console.log("Note: 'head vertical' commands will be accepted but filtered out.");

function promptUser() {
    rl.question('Command: ', (input) => {
        handleInput(input);
        promptUser(); 
    });
}

function handleInput(input) {
    const parts = input.trim().split(/\s+/);
    
    // Extract degrees
    const degreesStr = parts.pop();
    const degrees = parseInt(degreesStr);
    
    // Extract name
    const servoName = parts.join(' ').toLowerCase();

    if (isNaN(degrees)) {
        console.log("Error: Invalid degree value.");
        return;
    }

    if (!servoMap.hasOwnProperty(servoName)) {
        console.log(`Error: Unknown servo name '${servoName}'.`);
        return;
    }

    const targetIndex = servoMap[servoName];

    // 1. Create array of 9 servos (Raw Indices 0-8)
    // Default all to 90 degrees
    let rawPose = new Array(9).fill(90);

    // 2. Set the specific servo to the requested degrees
    rawPose[targetIndex] = degrees;

    // 3. Format & Filter data
    // We iterate 0-8, but SKIP index 1 exactly like your backend logic:
    // "if (i !== 1 && i < 9)"
    let sendData = "";
    
    for (let i = 0; i < rawPose.length; i++) {
        // EXCLUSION LOGIC: Skip Head Vertical (Index 1)
        if (i === 1) continue;

        // Process data
        let reducedValue = Math.floor(rawPose[i] / 10);
        
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
        console.log(`Sent: ${sendData} (Length: ${sendData.length})`);
        if (targetIndex === 1) {
            console.log("Warning: You moved Head Vertical, but it was filtered out of the serial data.");
        }
    });
}

promptUser();