import cv2
import numpy as np

# The "calculate_areas_and_display" function processes an input image containing four equally spaced colored bars. 
# It calculates the area of each bar, visualizes the areas by labeling them at the bar centroids, and displays the modified image 
# using OpenCV. The function uses pixel counting to estimate bar areas and calculates centroid positions for labeling. 
# After displaying the image with area information, it waits for a key press before closing the display window. 
# This function provides a simple yet informative analysis of colored bars in an image.
def calculate_areas_and_display(image_path):

    image = cv2.imread(image_path)
    
    height, width, _ = image.shape
    
    bar_width = width // 4
    
    areas = []
    for i in range(4):
        x_start = i * bar_width
        x_end = (i + 1) * bar_width
        
        roi = image[0:height, x_start:x_end]
        
        area = np.count_nonzero(roi)
        areas.append(area)
        
        centroid_x = (x_start + x_end) // 2
        centroid_y = height // 2
        
        cv2.putText(image, f" {area} px^2", (centroid_x - 30, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    
    cv2.imshow("Colored Bars Analysis", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

input_image_path = "Images\\fig1.jpg"

calculate_areas_and_display(input_image_path)
