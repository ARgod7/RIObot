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

def _sweep_elbows(
    port,
    rsv,
    lsv,
    rsh,
    lsh,
    re_a,
    le_a,
    re_b,
    le_b,
    steps,
    step_delay,
):
    """Linearly interpolate elbows from (re_a, le_a) to (re_b, le_b)."""
    for i in range(steps + 1):
        t = i / steps
        re = int(round(re_a + (re_b - re_a) * t))
        le = int(round(le_a + (le_b - le_a) * t))
        send_pose(port, rsv=rsv, lsv=lsv, rsh=rsh, lsh=lsh, re=re, le=le)
        time.sleep(step_delay)


def smooth_wave(port, cycles=0, step_delay=0.03, sweep_steps=30):
    """
    Make both arms wave. Elbows sweep smoothly; motion repeats until stopped.

    Right arm: RSV=150, RE: 90->0->90
    Left arm: LSV=30, LE: 90->180->90

    Args:
        port: Serial port object
        cycles: Number of full wave cycles; 0 = run continuously until Ctrl+C
        step_delay: Seconds between interpolated elbow steps
        sweep_steps: How many steps per elbow sweep (higher = smoother, slower)
    """
    continuous = cycles == 0
    if continuous:
        print("Continuous wave — Ctrl+C to stop")
    else:
        print(f"Making wave motion for {cycles} cycles")
    print("Right arm: RSV=150°, RE: 90°->0°->90°")
    print("Left arm: LSV=30°, LE: 90°->180°->90°")
    print(f"Step delay: {step_delay}s ({sweep_steps} steps per sweep)\n")

    try:
        print("Positioning shoulders...")
        send_pose(port, rsv=150, lsv=30, rsh=90, lsh=90, re=90, le=90)
        time.sleep(1.5)

        print("Starting wave motion!\n")

        cycle = 0
        while True:
            cycle += 1
            if continuous:
                print(f"Wave cycle {cycle}")
            else:
                print(f"Wave {cycle}/{cycles}")

            _sweep_elbows(
                port, 150, 30, 90, 90, 90, 90, 0, 180, sweep_steps, step_delay
            )
            _sweep_elbows(
                port, 150, 30, 90, 90, 0, 180, 90, 90, sweep_steps, step_delay
            )

            if not continuous and cycle >= cycles:
                break

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
    
    print("Both arms wave with smooth elbow sweeps (repeats until you stop).")
    print("Right: shoulder 150°, elbow 90°->0°->90°. Left: shoulder 30°, elbow 90°->180°->90°.")
    print("Default is continuous (0 cycles); use Ctrl+C to stop.\n")
    
    # Get user input (0 cycles = continuous until Ctrl+C)
    try:
        cycles = int(
            input("Number of wave cycles (0 = continuous; default 0): ") or "0"
        )
        step_delay = float(
            input("Delay per sweep step in seconds (default 0.03): ") or "0.03"
        )
    except ValueError:
        print("Invalid input, using defaults")
        cycles = 0
        step_delay = 0.03

    print()

    smooth_wave(port, cycles=cycles, step_delay=step_delay)
    
    # Close port
    port.close()
    print("Port closed. Exiting.")

if __name__ == "__main__":
    main()