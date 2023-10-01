import cv2

# The classify_gender function in the code performs gender classification of characters in cartoon images. 
# It utilizes template matching to compare predefined boy and girl templates with a left cartoon image. 
# After calculating similarity scores, it applies a matching threshold to determine whether the character in the left image is a boy 
# or a girl. The function then returns the classification result, providing insight into the gender of the character. 
# It can be a useful tool for automating gender identification in cartoon imagery, with the flexibility to adjust templates 
# and thresholds for different images.
def classify_gender(left_image_path, right_image_path):
    
    boy_template = cv2.imread("Images\\fig3.jpg", cv2.IMREAD_GRAYSCALE)
    girl_template = cv2.imread("Images\\fig4.jpg", cv2.IMREAD_GRAYSCALE)

    left_image = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)

    if boy_template is None or girl_template is None or left_image is None or right_image is None:
        return "Error: Could not load one or both of the images or templates."

    matching_threshold = 0.8 

    boy_match = cv2.matchTemplate(left_image, boy_template, cv2.TM_CCOEFF_NORMED)
    girl_match = cv2.matchTemplate(left_image, girl_template, cv2.TM_CCOEFF_NORMED)

    boy_score = cv2.minMaxLoc(boy_match)[1]
    girl_score = cv2.minMaxLoc(girl_match)[1]

    if boy_score > matching_threshold and girl_score <= matching_threshold:
        result = "Left image is classified as a boy, and right image is classified as a girl."
    elif boy_score <= matching_threshold and girl_score > matching_threshold:
        result = "Left image is classified as a girl, and right image is classified as a boy."
    else:
        result = "Cannot determine based on template matching. Please try different templates or adjust the threshold."

    # Display the left and right images
    cv2.imshow("Left Image", left_image)
    cv2.imshow("Right Image", right_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result

left_image_path = "Images\\fig3.jpg"
right_image_path = "Images\\fig4.jpg"

result = classify_gender(left_image_path, right_image_path)
print(result)
