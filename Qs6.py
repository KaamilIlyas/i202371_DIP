import cv2
import numpy as np

#The `separate_bone_segments` function processes an input image with segmented bone regions. 
# It first converts the image to the HSV color space and defines color bounds for each segment. 
# The function then identifies valid segments, calculates their maximum dimensions, and draws bounding boxes around them. 
# Extracted segments are stored and returned. Example usage displays these segmented regions. 
# This function aids in bone region analysis by segmenting and visualizing bone regions in an image.
def separate_bone_segments(image_path):

    image = cv2.imread(image_path)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_bounds = [(97, 50, 50), (-10,  50,  50), (49, 50, 50), (81, 50, 50), (3, 50, 50)]
    upper_bounds = [(137, 255, 255), (30, 255, 255), (89, 255, 255), (121, 255, 255), (43, 255, 255)]

    bone_segments = []

    for i, (lower, upper) in enumerate(zip(lower_bounds, upper_bounds)):

        mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]

        max_width = 0
        max_height = 0

        for contour in valid_contours:
            x, y, w, h = cv2.boundingRect(contour)

            if w > max_width:
                max_width = w
            if h > max_height:
                max_height = h

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            segment = image[y:y+h, x:x+w]
            bone_segments.append(segment)

        print(f"Segment {i + 1}:")
        print(f"Maximum Width: {max_width}px")
        print(f"Maximum Height: {max_height}px")
        print()

    return bone_segments

image_path = "Images\\finger-bones.jpg" 
segments = separate_bone_segments(image_path)

for i, segment in enumerate(segments):
    cv2.imshow(f"Segment {i + 1}", segment)

cv2.waitKey(0)
cv2.destroyAllWindows()
