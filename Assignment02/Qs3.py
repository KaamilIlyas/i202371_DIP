import cv2
import numpy as np
import matplotlib.pyplot as plt

# It computes a score based on the filled pixel percentage in a region of interest (ROI) by first converting the ROI to grayscale. 
# It determines the ratio of filled pixels to the total pixels in the ROI and derives a score by dividing the percentage by 10. 
# This score reflects the "filledness" of the character in the ROI.
def calculate_score(roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    filled_pixels = cv2.countNonZero(gray_roi)
    total_pixels = gray_roi.shape[0] * gray_roi.shape[1]
    
    percentage_filled = (filled_pixels / total_pixels) * 100
    
    score = int(percentage_filled / 10)
    
    return score

# This function processes an input image, extracting the three largest character contours. It calculates scores for each character 
# using the calculate_score function and annotates the image with these scores. The annotated image is displayed using matplotlib. 
# This code is useful for assessing handwritten character quality and can be applied in OCR and character evaluation scenarios.
def process_and_display_scores(input_image_path):

    input_image = cv2.imread(input_image_path)

    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_image, threshold1=0.4, threshold2=0.9)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    scores = []

    image_with_scores = input_image.copy()

    for idx, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        roi = input_image[y:y + h, x:x + w]

        score = calculate_score(roi)
        scores.append(score)

        text_x = x + w // 2 - 20 
        text_y = y - 10

        cv2.putText(image_with_scores, f"Score: {score}", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    plt.imshow(cv2.cvtColor(image_with_scores, cv2.COLOR_BGR2RGB))
    plt.title("Image with Scores")
    plt.axis('off')
    plt.show()

process_and_display_scores("Assignment02\\Images\\3-1.jpg")
