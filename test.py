import serial
import sys

# --- Configuration ---
COM_PORT = 5
BAUD_RATE = 9600

# --- Servo Mapping ---
# Maps user-friendly names to the index in the *sent* data array (0-7)
servo_map = {
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
}

# --- Helper: Print Available Names ---
def print_servo_list():
    print("\n--- Available Servo Names ---")
    
    # Reverse the map to group aliases by their ID (0-7)
    grouped_names = {}
    for name, index in servo_map.items():
        if index not in grouped_names:
            grouped_names[index] = []
        grouped_names[index].append(name)
    
    # Print each group
    for servo_id in sorted(grouped_names.keys()):
        print(f"ID {servo_id}: [ {', '.join(grouped_names[servo_id])} ]")
    print("-----------------------------\n")

# --- Main Logic ---
def main():
    # Setup Serial Port
    try:
        port = serial.Serial(f'COM{COM_PORT}', BAUD_RATE, timeout=1)
        print(f"--- Servo Tester Initialized on COM{COM_PORT} ---")
    except serial.SerialException as e:
        print(f"Error opening port: {e}")
        sys.exit(1)
    
    # Print the servo names
    print_servo_list()
    
    print("Enter command in format: <servo_name> <degrees>")
    print("Examples: 'head 45', 'base 120', 'rsv 180'")
    print("All other servos will reset to 90 degrees.")
    print()
    
    # Main input loop
    while True:
        try:
            user_input = input("Command: ")
            handle_input(user_input, port)
        except KeyboardInterrupt:
            print("\nExiting...")
            port.close()
            sys.exit(0)
        except EOFError:
            print("\nExiting...")
            port.close()
            sys.exit(0)

def handle_input(user_input, port):
    parts = user_input.strip().split()
    
    if not parts:
        return
    
    # Extract degrees (last part of the string)
    degrees_str = parts[-1]
    try:
        degrees = int(degrees_str)
    except ValueError:
        print("Error: Invalid degree value. Format: name degrees")
        return
    
    # Rejoin the rest to get the name
    servo_name = ' '.join(parts[:-1]).lower()
    
    if servo_name not in servo_map:
        print(f"Error: Unknown servo name '{servo_name}'. Please check the list above.")
        return
    
    target_index = servo_map[servo_name]
    
    # 1. Create array of 8 servos (indices 0-7), defaulting all to 90 degrees
    current_pose = [90] * 8
    
    # 2. Set the specific servo to the requested degrees
    current_pose[target_index] = degrees
    
    # 3. Format data according to your backend protocol
    send_data = ""
    
    for value in current_pose:
        # Divide by 10 and floor
        reduced_value = value // 10
        
        # Pad to 2 digits (e.g. 5 -> "05")
        send_data += f"{reduced_value:02d}"
    
    # 4. Send over Serial
    try:
        port.write(send_data.encode())
        print(f"Sent: {send_data} (Moved '{servo_name}' to {degrees}°)")
    except serial.SerialException as e:
        print(f"Error on write: {e}")

if __name__ == "__main__":
    main()