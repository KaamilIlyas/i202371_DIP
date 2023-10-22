import cv2
import numpy as np

# The code defines a function "detect_clock_time(image_path)" to determine the current time displayed on an analog clock image. 
# It processes the input image by extracting contours representing the clock's hands, fitting ellipses to them, and calculating their angles. 
# These angles are used to derive the hour and minute, adjusted for a specific clock time. The code prints the calculated time and 
# displays the image with detected clock hands.
def detect_clock_time(image_path):
    image = cv2.imread(image_path)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    angle = 0.0

    for contour in contours:

        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            (center, axes, angle) = ellipse

            cv2.ellipse(image, (int(center[0]), int(center[1])),
                    (int(axes[0] / 2), int(axes[1] / 2)), angle, 0, 360, (0, 0, 255), 2)

    hour_angle = (angle + 120) % 360
    minute_angle = (angle + 180) % 360 

    hour = int((hour_angle / 360) * 12)
    minute = int((minute_angle / 360) * 60)

    print(f"Hour hand angle: {hour_angle} degrees, Minute hand angle: {minute_angle} degrees")
    print(f"Current Time: {hour:02d}:{minute:02d}")

    cv2.imshow('Clock with Hands Detected', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

detect_clock_time('Assignment02\\Images//2-2.jpg')
