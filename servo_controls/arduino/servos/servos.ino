// Pose Imitator Arduino Mega Code (Serial Ears Version)
// Debasish Buragohain
#include <Servo.h>

Servo servoHeadHor;
Servo servoHeadVer;
Servo servorightShoulderY;
Servo servoleftShoulderY;
Servo servorightShoulderX;
Servo servoleftShoulderX;
Servo servorightElbow;
Servo servoleftElbow;
Servo servobase;
Servo servorightEar;
Servo servoleftEar;

void setup() {
  Serial.begin(9600);
  Serial3.begin(9600);
  Serial.print("Servo controller ready.");
  Serial.println("\n");
  pinMode(LED_BUILTIN, OUTPUT);

  servoleftElbow.attach(48);
  servoleftShoulderY.attach(46);
  servoleftShoulderX.attach(44);
  servorightElbow.attach(42);
  servorightShoulderY.attach(40);
  servorightShoulderX.attach(38);
  servobase.attach(36);
  servoHeadVer.attach(34);
  servoHeadHor.attach(32);
  servoleftEar.attach(52);
  servorightEar.attach(50);

  delay(1000);
  servoHeadHor.write(90);
  servoHeadVer.write(90);
  servorightShoulderY.write(90);
  servoleftShoulderY.write(90);
  servorightShoulderX.write(0);
  servoleftShoulderX.write(180);
  servorightElbow.write(90);
  servoleftElbow.write(90);
  servobase.write(90);
  servorightEar.write(90);
  servoleftEar.write(90);
  delay(500);
}

void loop() {
  static unsigned long lastOnlineMessage = 0;

  if (millis() - lastOnlineMessage >= 3000) {
    Serial3.println("online");
    lastOnlineMessage = millis();
  }

  if (Serial3.available()) {
    String inputText;
    while (Serial3.available()) {
      char c = Serial3.read();
      delay(10);
      inputText += c;
    }

    // 11 servos * 2 digits each = 22 characters.
    if (inputText.length() > 22) {
      inputText.remove(22);
    }

    Serial.println("");
    Serial.print("Raw Input: ");
    Serial.println(inputText);

    if (inputText.length() != 22) {
      Serial.print("Invalid length: ");
      Serial.print(inputText.length());
      for (int dh = 0; dh < 2; dh++) {
        digitalWrite(LED_BUILTIN, HIGH); delay(100);
        digitalWrite(LED_BUILTIN, LOW); delay(100);
      }
      return;
    }
    else {
      int inputDegrees[11] = { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 };

      for (int i = 0; i <= 21; i += 2) {
        int tens = (int(inputText[i]) - 48) * 10;
        int ones = (int(inputText[i + 1]) - 48);
        inputDegrees[i / 2] = (tens + ones) * 10;
      }

      String displayIt;
      for (int i = 0; i < 11; i++) {
        if (inputDegrees[i] == -1) {
          Serial.print("Error: input degree array contains an error value.");
          return;
        }
        displayIt += String(inputDegrees[i]) + ' ';
      }
      Serial.print("Parsed Angles: ");
      Serial.println(displayIt);

      // Existing mapping retained from the source file.
      if (servoHeadHor.read() != inputDegrees[0]) { servoHeadHor.write(inputDegrees[0]); delay(150); }
      if (servorightShoulderY.read() != inputDegrees[1]) { servorightShoulderY.write(inputDegrees[1]); delay(150); }
      if (servoleftShoulderY.read() != inputDegrees[2]) { servoleftShoulderY.write(inputDegrees[2]); delay(150); }
      if (servorightShoulderX.read() != inputDegrees[3]) { servorightShoulderX.write(inputDegrees[3]); delay(150); }
      if (servoleftShoulderX.read() != inputDegrees[4]) { servoleftShoulderX.write(inputDegrees[4]); delay(150); }
      if (servorightElbow.read() != inputDegrees[5]) { servorightElbow.write(inputDegrees[5]); delay(150); }
      if (servoleftElbow.read() != inputDegrees[6]) { servoleftElbow.write(inputDegrees[6]); delay(150); }
      if (servobase.read() != inputDegrees[7]) { servobase.write(inputDegrees[7]); delay(150); }

      // Use slot 8 for head vertical so all 11 serial values are applied.
      if (servoHeadVer.read() != inputDegrees[8]) { servoHeadVer.write(inputDegrees[8]); delay(150); }

      // New serial-controlled ear slots.
      // Last second value -> left ear, last value -> right ear.
      if (servoleftEar.read() != inputDegrees[9]) { servoleftEar.write(inputDegrees[9]); delay(150); }
      if (servorightEar.read() != inputDegrees[10]) { servorightEar.write(inputDegrees[10]); delay(150); }
    }
    delay(40);
  }
}
