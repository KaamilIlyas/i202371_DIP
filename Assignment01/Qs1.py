import cv2
import numpy as np

image = cv2.imread("Images\\rect1.jpg")

# Function for Grayscale Conversion (Intensity Transformation)
# The function takes an input color image and converts it to grayscale using the cv2.cvtColor function
# It returns the resulting grayscale image for further processing. Grayscale conversion simplifies intensity-based analysis
def convert_to_grayscale(input_image):
    return cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# This function performs image segmentation to identify and extract contours from the input image
# It first converts the input image to grayscale using the previously defined function
# Then, it applies binary thresholding to create a binary image where objects are separated from the background
# Finally, it detects and returns a list of detected contours using the cv2.findContours function
# The detected contours represent the boundaries of objects or regions in the image
def segment_image(input_image):
    gray_image = convert_to_grayscale(input_image)
    _, binary_thresholded = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# This function analyzes the shape represented by a given contour
# It calculates an approximate polygon for the contour and checks if it has four vertices, indicating a potential rectangle or square
# If it identifies a four-sided polygon, it calculates the perimeter, centroid coordinates, and aspect ratio of the bounding rectangle
# Depending on the aspect ratio, it categorizes the shape as a "Square," "Rectangle," or "Neither Square nor Rectangle."
# The function returns a tuple containing the shape type, perimeter, and centroid coordinates for further analysis and printing
def analyze_shape(contour):
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 4:
        perimeter = cv2.arcLength(contour, True)
        M = cv2.moments(contour)
        centroid_x = int(M["m10"] / M["m00"])
        centroid_y = int(M["m01"] / M["m00"])
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h

        if 0.9 <= aspect_ratio <= 1.1:
            shape_type = "Square"
        elif 1.5 <= aspect_ratio <= 2.5:
            shape_type = "Rectangle"
        else:
            shape_type = "Neither Square nor Rectangle"
        
        return shape_type, perimeter, (centroid_x, centroid_y)

detected_contours = segment_image(image)

for contour in detected_contours:
    shape_info = analyze_shape(contour)
    if shape_info:
        shape_type, perimeter, centroid = shape_info
        print(f"Shape Type: {shape_type}")
        print(f"Perimeter: {perimeter}")
        print(f"Centroid: {centroid}")

cv2.drawContours(image, detected_contours, -1, (0, 255, 0), 2)
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()