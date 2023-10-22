import cv2

# The function analyzes two grayscale images by calculating their Laplacian variance, a measure of intensity transitions. 
# It uses a defined `gradient_threshold` to classify one as "Blurred Image" if its variance is below the threshold and the other as
# "Original Image" if its variance exceeds the threshold. The function returns titles indicating these classifications, 
# enabling visual confirmation of which image is original and which is blurred. This method offers a straightforward approach to 
# differentiate between original and blurred images based on gradient characteristics.
def detect_blurred_image(original_image, blurred_image):
    
    original_var = cv2.Laplacian(original_image, cv2.CV_64F).var()
    blurred_var = cv2.Laplacian(blurred_image, cv2.CV_64F).var()

    gradient_threshold = 100  

    if blurred_var < gradient_threshold:
        return "Blurred Image", "Original Image"
    else:
        return "Original Image", "Blurred Image"
    
original_image = cv2.imread("Images\\fig5.jpg", cv2.IMREAD_GRAYSCALE)
blurred_image = cv2.imread("Images\\fig5_blur.jpg", cv2.IMREAD_GRAYSCALE)

if original_image is None or blurred_image is None:
    print("Error: Could not load one or both of the images.")
else:
    blurred_title, original_title = detect_blurred_image(original_image, blurred_image)

    cv2.imshow(blurred_title, blurred_image)
    cv2.imshow(original_title, original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()