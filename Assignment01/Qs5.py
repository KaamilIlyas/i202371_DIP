import cv2
import numpy as np

image = cv2.imread("Images\\fig2.jpg")

image_bgr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

yellow_lower = (233, 198, 78)
yellow_upper = (273, 238, 118)
light_gray_lower = (211, 209, 210)
light_gray_upper = (251, 249, 250)
gray_lower = (197, 197, 197)
gray_upper = (237, 237, 237)
dark_gray_lower = (154, 150, 149)
dark_gray_upper = (194, 190, 189)

total_pixels = image_bgr.shape[0] * image_bgr.shape[1]

# The `calculate_percentage_area` function determines the percentage area of a specific color in an image. 
# It takes the lower and upper color bounds as inputs, creates a mask to isolate pixels within the color range, 
# counts non-zero pixels in the mask, and calculates the percentage area relative to the total image size. 
# This function is utilized to find the coverage percentages of different colors (yellow, light gray, gray, dark gray) 
# in the loaded image, and the results are printed in the specified format.
def calculate_percentage_area(lower_bound, upper_bound):
    mask = cv2.inRange(image_bgr, lower_bound, upper_bound)
    area = (np.count_nonzero(mask) / total_pixels) * 100
    return area

area_yellow = calculate_percentage_area(yellow_lower, yellow_upper)
area_light_gray = calculate_percentage_area(light_gray_lower, light_gray_upper)
area_gray = calculate_percentage_area(gray_lower, gray_upper)
area_dark_gray = calculate_percentage_area(dark_gray_lower, dark_gray_upper)

print("Bar Area Covered(%)")
print(f"Yellow: {area_yellow:.0f}%")
print(f"Light Gray: {area_light_gray:.0f}%")
print(f"Gray: {area_gray:.0f}%")
print(f"Dark Gray: {area_dark_gray:.0f}%")