import cv2
import numpy as np

input_image = cv2.imread("Assignment02\\Images\\1-3.jpg", cv2.IMREAD_COLOR)

character_templates = {
    'A': cv2.imread("Assignment02\\Images\\A.png", cv2.IMREAD_COLOR),
    '8': cv2.imread("Assignment02\\Images\\8.png", cv2.IMREAD_COLOR),
    'W': cv2.imread("Assignment02\\Images\\W.png", cv2.IMREAD_COLOR)
}

# This function identifies a character within a specified input region. It iterates through a dictionary of character templates, 
# which are images of characters to be recognized. The function compares the input region with these templates using template matching 
# (cv2.matchTemplate) and selects the character with the highest similarity (maximum match value) as the identified character.
def identify_input_character(input_region):
    max_match_score = 0
    identified_character = None

    for char, template in character_templates.items():
        match_result = cv2.matchTemplate(input_region, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)

        if max_val > max_match_score:
            max_match_score = max_val
            identified_character = char

    return identified_character

# This function draws bounding boxes around the matching characters in the lower section of the image. It takes the lower portion 
# of the image and the character to be matched as inputs. For each matching location found in the lower section, it draws a green 
# rectangle around the recognized character. The template matching is performed with a specified threshold for similarity to identify 
# valid character matches.
def draw_matching_characters(lower_section, character_to_match):
    template = character_templates.get(character_to_match, None)

    if template is not None:
        match_result = cv2.matchTemplate(lower_section, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.95 

        loc = np.where(match_result >= threshold)
        if loc[0].size > 0:  
            for pt in zip(*loc[::-1]):
                cv2.rectangle(lower_section, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0, 255, 0), 2)
        else:
            print("No match found")

input_character = identify_input_character(input_image[:input_image.shape[0] // 2, :])

if input_character:

    draw_matching_characters(input_image[input_image.shape[0] // 2 :, :], input_character)
    cv2.imshow("Image", input_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
