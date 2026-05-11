"""
test_emotion_dynamics.py
------------------------
Test that emotion vectors are updating in real-time and not frozen.

Run this to verify:
1. Face detector is continuously detecting emotions
2. Posture detector is continuously detecting emotions
3. Fusion is updating in real-time
4. Smoothing is not causing stagnation
"""

import time
import logging
from perception.perception_loop import PerceptionLoop
from perception.emotion_fusion import EmotionVector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def test_emotion_dynamics():
    """Test that emotions update continuously."""
    print("\n" + "="*80)
    print("EMOTION DYNAMICS TEST - Verifying Real-Time Updates")
    print("="*80 + "\n")

    loop = PerceptionLoop(
        enable_voice=False,  # Disable voice for faster testing
        enable_face=True,
        enable_posture=True,
    )

    print("🎬 Starting perception loop...")
    loop.start()
    loop.wait_until_ready(timeout=10)

    print("📊 Monitoring emotion updates for 20 seconds...\n")

    last_emotions = None
    update_count = 0
    change_count = 0

    try:
        for i in range(20):
            time.sleep(1)

            # Get current fused emotions
            fused = loop.get_fused()
            current_emotions = fused.to_dict()

            # Check for changes
            if last_emotions is not None:
                # Calculate max change across all emotions
                max_change = max(
                    abs(current_emotions[e] - last_emotions[e])
                    for e in current_emotions
                )

                # Show status
                if max_change > 0.01:  # Threshold for "changed"
                    change_count += 1
                    change_indicator = "✅ UPDATING"
                else:
                    change_indicator = "❌ FROZEN"

                # Get dominant emotion
                dominant = max(current_emotions.items(), key=lambda x: x[1])[0]
                dominant_value = current_emotions[dominant]

                print(f"[{i+1:02d}s] {change_indicator:15} | {dominant:10} | "
                      f"Confidence: {fused.confidence:.2f} | Max Δ: {max_change:.3f}")

                update_count += 1
            else:
                print(f"[{i+1:02d}s] Initial reading")

            last_emotions = current_emotions

    except KeyboardInterrupt:
        print("\nTest interrupted by user.")

    finally:
        loop.stop()
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        if update_count > 0:
            update_rate = (change_count / update_count) * 100
            print(f"Total readings: {update_count}")
            print(f"Updates detected: {change_count}")
            print(f"Update rate: {update_rate:.1f}%")

            if update_rate > 50:
                print("\n✅ RESULT: Emotions are updating WELL")
                print("   Emotion vectors are changing in real-time as expected.")
            elif update_rate > 20:
                print("\n⚠️  RESULT: Emotions are updating SLOWLY")
                print("   Emotion vectors are updating but could be more responsive.")
                print("   Consider reducing smoothing alpha or checking detector sensitivity.")
            else:
                print("\n❌ RESULT: Emotions are FROZEN or barely updating")
                print("   Perception loop may not be running correctly.")
                print("   Check detectors are initialized properly.")
        else:
            print("❌ No readings obtained. Perception loop may have failed to start.")

        print("\n" + "="*80)

if __name__ == "__main__":
    test_emotion_dynamics()

