import serial
import sys
import time

# --- Configuration ---
COM_PORT = 5
BAUD_RATE = 9600

# --- Servo Mapping ---
# right shoulder vertical (rsv): 1
# left shoulder vertical (lsv): 2
# right shoulder horizontal (rsh): 3
# left shoulder horizontal (lsh): 4
# right elbow (re): 5
# left elbow (le): 6

def send_pose(port, rsv=90, lsv=90, rsh=90, lsh=90, re=90, le=90):
    """
    Send a complete pose with specific angles for arm servos
    All other servos default to 90 degrees
    
    Args:
        port: Serial port object
        rsv: Right shoulder vertical angle (0-180)
        lsv: Left shoulder vertical angle (0-180)
        rsh: Right shoulder horizontal angle (0-180)
        lsh: Left shoulder horizontal angle (0-180)
        re: Right elbow angle (0-180)
        le: Left elbow angle (0-180)
    """
    # Create array of 8 servos, all at 90 degrees
    pose = [90] * 8
    
    # Set arm servos
    pose[1] = rsv  # right shoulder vertical
    pose[2] = lsv  # left shoulder vertical
    pose[3] = rsh  # right shoulder horizontal
    pose[4] = lsh  # left shoulder horizontal
    pose[5] = re   # right elbow
    pose[6] = le   # left elbow
    
    # Format data according to protocol
    send_data = ""
    for value in pose:
        reduced_value = value // 10
        send_data += f"{reduced_value:02d}"
    
    # Send over Serial
    try:
        port.write(send_data.encode())
        return True
    except serial.SerialException as e:
        print(f"Error on write: {e}")
        return False

def smooth_wave(port, cycles=3, delay=1.0):
    """
    Make both arms wave
    Right arm: RSV=150, RE: 90->0->90
    Left arm: LSV=30, LE: 90->180->90
    
    Args:
        port: Serial port object
        cycles: Number of wave cycles (default 3)
        delay: Time between poses in seconds (default 1.0)
    """
    print(f"Making wave motion for {cycles} cycles")
    print("Right arm: RSV=150°, RE: 90°->0°->90°")
    print("Left arm: LSV=30°, LE: 90°->180°->90°")
    print(f"Delay: {delay} seconds between poses")
    print("Press Ctrl+C to stop\n")
    
    try:
        # Move shoulders to position with elbows at 90
        print("Positioning shoulders...")
        send_pose(port, rsv=150, lsv=30, rsh=90, lsh=90, re=90, le=90)
        time.sleep(1.5)
        
        print(f"\nStarting wave motion!\n")
        
        for cycle in range(cycles):
            print(f"Wave {cycle + 1}/{cycles}")
            
            # Pose 1: Elbows outward
            send_pose(port, rsv=150, lsv=30, rsh=90, lsh=90, re=0, le=180)
            time.sleep(delay)
            
            # Pose 2: Elbows back to center
            send_pose(port, rsv=150, lsv=30, rsh=90, lsh=90, re=90, le=90)
            time.sleep(delay)
        
        # Return to neutral position
        print("\nReturning to neutral position...")
        send_pose(port, rsv=90, lsv=90, rsh=90, lsh=90, re=90, le=90)
        time.sleep(1)
        
        print("Wave complete!")
        
    except KeyboardInterrupt:
        print("\n\nWave interrupted! Returning to neutral position...")
        send_pose(port, rsv=90, lsv=90, rsh=90, lsh=90, re=90, le=90)

def main():
    # Setup Serial Port
    try:
        port = serial.Serial(f'COM{COM_PORT}', BAUD_RATE, timeout=1)
        print(f"--- Robot Smooth Wave Controller on COM{COM_PORT} ---\n")
    except serial.SerialException as e:
        print(f"Error opening port: {e}")
        sys.exit(1)
    
    print("This will make both arms wave smoothly and simultaneously")
    print("Right arm: Shoulder up (150°), elbow sweeps 90°->0°->90°")
    print("Left arm: Shoulder down (30°), elbow sweeps 90°->180°->90°\n")
    
    # Get user input
    try:
        cycles = int(input("Number of wave cycles (default 3): ") or "3")
        delay = float(input("Delay between poses in seconds (default 1.0): ") or "1.0")
    except ValueError:
        print("Invalid input, using defaults")
        cycles = 3
        delay = 1.0
    
    print()
    
    # Perform the wave
    smooth_wave(port, cycles=cycles, delay=delay)
    
    # Close port
    port.close()
    print("Port closed. Exiting.")

if __name__ == "__main__":
    main()