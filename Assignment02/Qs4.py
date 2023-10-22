import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import math

root = tk.Tk()
root.title("Image Processing Software")

loaded_image = None

# Opens an image using a file dialog and displays it in the software's canvas.
def open_image():
    global loaded_image
    file_path = filedialog.askopenfilename()
    if file_path:
        loaded_image = cv2.imread(file_path)
        display_image(loaded_image)
        enable_save_button() 

def display_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    photo = ImageTk.PhotoImage(image)
    canvas.create_image(0, 0, anchor=tk.NW, image=photo)
    canvas.image = photo

# Saves the provided image using a file dialog.
def save_image(image_to_save):
    if image_to_save is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".png")
        if file_path:
            cv2.imwrite(file_path, image_to_save)

# Enables the "Save Image" button for saving the modified image.
def enable_save_button():
    save_button["state"] = "normal"
    save_button["command"] = lambda: save_image(loaded_image)

# Adjusts the brightness of the loaded image based on the specified brightness value.
def adjust_brightness(brightness_value):
    global loaded_image
    if loaded_image is not None:
        brightness = int(brightness_value)
        adjusted_image = cv2.addWeighted(loaded_image, 1 + (brightness / 100), loaded_image, 0, 0)
        display_image(adjusted_image)
        enable_save_button() 

# Applies a logarithmic transformation to enhance image contrast.
def apply_logarithm():
    global loaded_image
    if loaded_image is not None:
        c = 255 / math.log(1 + loaded_image.max())
        log_image = c * (np.log(loaded_image.astype(np.float32) + 1))
        log_image = cv2.convertScaleAbs(log_image)
        display_image(log_image)
        enable_save_button()  

# Applies binary thresholding to the image based on the specified threshold value.
def apply_threshold(threshold_value):
    global loaded_image
    if loaded_image is not None:
        _, thresholded_image = cv2.threshold(loaded_image, threshold_value, 255, cv2.THRESH_BINARY)
        display_image(thresholded_image)
        enable_save_button() 

# Converts the image to grayscale.
def convert_to_grayscale():
    global loaded_image
    if loaded_image is not None:
        grayscale_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
        display_image(grayscale_image)
        enable_save_button()  

# Applies various filters, such as Gaussian, Median, or Bilateral, to the image.
def apply_filter(filter_type):
    global loaded_image
    if loaded_image is not None:
        if filter_type == "Gaussian":
            filtered_image = cv2.GaussianBlur(loaded_image, (5, 5), 0)
        elif filter_type == "Median":
            filtered_image = cv2.medianBlur(loaded_image, 5)
        elif filter_type == "Bilateral":
            filtered_image = cv2.bilateralFilter(loaded_image, 9, 75, 75)
        display_image(filtered_image)
        enable_save_button() 

# Enhances image edges using the Laplacian filter.
def apply_laplacian_sharpening():
    global loaded_image
    if loaded_image is not None:
        laplacian_image = cv2.Laplacian(loaded_image, cv2.CV_64F)
        laplacian_image = cv2.convertScaleAbs(laplacian_image)
        sharpened_image = cv2.addWeighted(loaded_image, 1.5, laplacian_image, -0.5, 0)
        display_image(sharpened_image)
        enable_save_button() 

# Applies unsharp masking to enhance image details and edges.
def apply_unsharp_masking():
    global loaded_image
    if loaded_image is not None:
        blurred_image = cv2.GaussianBlur(loaded_image, (0, 0), 3)
        unsharp_masked_image = cv2.addWeighted(loaded_image, 1.5, blurred_image, -0.5, 0)
        display_image(unsharp_masked_image)
        enable_save_button() 

# Performs morphological operations, such as Erosion, Dilation, Opening, or Closing, on the image.
def apply_morphological_operation(operation):
    global loaded_image
    if loaded_image is not None:
        kernel = np.ones((5, 5), np.uint8)
        if operation == "Erode":
            morph_image = cv2.erode(loaded_image, kernel, iterations=1)
        elif operation == "Dilate":
            morph_image = cv2.dilate(loaded_image, kernel, iterations=1)
        elif operation == "Open":
            morph_image = cv2.morphologyEx(loaded_image, cv2.MORPH_OPEN, kernel)
        elif operation == "Close":
            morph_image = cv2.morphologyEx(loaded_image, cv2.MORPH_CLOSE, kernel)
        display_image(morph_image)
        enable_save_button() 

# Detects and highlights contours in the image.
def detect_and_display_contours():
    global loaded_image
    if loaded_image is not None:
        grayscale_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
        _, thresholded_image = cv2.threshold(grayscale_image, 128, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_image = cv2.drawContours(loaded_image.copy(), contours, -1, (0, 0, 255), 2)

        display_image(contour_image)
        enable_save_button()  

# Segments the image based on specified color ranges.
def apply_color_segmentation():
    global loaded_image
    if loaded_image is not None:
        lower_range = np.array([30, 150, 50]) 
        upper_range = np.array([80, 255, 255]) 

        hsv_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_image, lower_range, upper_range)
        result_image = cv2.bitwise_and(loaded_image, loaded_image, mask=mask)

        display_image(result_image)
        enable_save_button()

# Detects and highlights lines in the image using the Hough Line Transform.
def detect_lines():
            global loaded_image
            if loaded_image is not None:
                gray_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
                lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)

            if lines is not None:
                for line in lines:
                    rho, theta = line[0]
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a * rho
                    y0 = b * rho
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(loaded_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

            display_image(loaded_image)
            enable_save_button()

# Identifies and annotates connected components in the image using binary image analysis techniques.
def connected_component_analysis():
    global loaded_image
    if loaded_image is not None:
        gray_image = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY)
        labeled_image, num_labels = cv2.connectedComponents(binary_image)
        labeled_image = cv2.cvtColor(labeled_image, cv2.COLOR_GRAY2BGR)

        for label in range(1, num_labels):
            label_mask = np.uint8(labeled_image == label) * 255
            contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area = cv2.contourArea(contours[0])
            perimeter = cv2.arcLength(contours[0], True)

            cv2.drawContours(labeled_image, contours, -1, (0, 0, 255), 2)
            cv2.putText(labeled_image, f"Object {label}", (contours[0][0][0][0], contours[0][0][0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(labeled_image, f"Area: {area}", (contours[0][0][0][0], contours[0][0][0][1] + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(labeled_image, f"Perimeter: {perimeter}", (contours[0][0][0][0], contours[0][0][0][1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        display_image(labeled_image)
        enable_save_button()

top_buttons_frame = tk.Frame(root)
top_buttons_frame.pack(side=tk.TOP, fill=tk.X)

open_button = tk.Button(top_buttons_frame, text="Open Image", command=open_image)
brightness_label = tk.Label(top_buttons_frame, text="Brightness:")
brightness_slider = tk.Scale(top_buttons_frame, from_=-100, to=100, orient="horizontal", resolution=1)
brightness_button = tk.Button(top_buttons_frame, text="Adjust Brightness", command=lambda: adjust_brightness(brightness_slider.get()))
save_button = tk.Button(top_buttons_frame, text="Save Image", state="disabled")
log_button = tk.Button(top_buttons_frame, text="Apply Logarithm", command=apply_logarithm)
grayscale_button = tk.Button(top_buttons_frame, text="Convert to Grayscale", command=convert_to_grayscale)
threshold_label = tk.Label(top_buttons_frame, text="Threshold:")
threshold_slider = tk.Scale(top_buttons_frame, from_=0, to=255, orient="horizontal", resolution=1)
threshold_button = tk.Button(top_buttons_frame, text="Thresholding", command=lambda: apply_threshold(threshold_slider.get()))
gaussian_button = tk.Button(top_buttons_frame, text="Gaussian Blur", command=lambda: apply_filter("Gaussian"))
median_button = tk.Button(top_buttons_frame, text="Median Filter", command=lambda: apply_filter("Median"))
bilateral_button = tk.Button(top_buttons_frame, text="Bilateral Filter", command=lambda: apply_filter("Bilateral"))
laplacian_button = tk.Button(top_buttons_frame, text="Laplacian Sharpening", command=apply_laplacian_sharpening)
unsharp_masking_button = tk.Button(top_buttons_frame, text="Unsharp Masking", command=apply_unsharp_masking)

open_button.pack(side=tk.LEFT, padx=10)
brightness_label.pack(side=tk.LEFT, padx=10)
brightness_slider.pack(side=tk.LEFT, padx=10)
brightness_button.pack(side=tk.LEFT, padx=10)
save_button.pack(side=tk.LEFT, padx=10)
log_button.pack(side=tk.LEFT, padx=10)
grayscale_button.pack(side=tk.LEFT, padx=10)
threshold_label.pack(side=tk.LEFT, padx=10)
threshold_slider.pack(side=tk.LEFT, padx=10)
threshold_button.pack(side=tk.LEFT, padx=10)
gaussian_button.pack(side=tk.LEFT, padx=10)
median_button.pack(side=tk.LEFT, padx=10)
bilateral_button.pack(side=tk.LEFT, padx=10)
laplacian_button.pack(side=tk.LEFT, padx=10)
unsharp_masking_button.pack(side=tk.LEFT, padx=10)

morphological_buttons_frame = tk.Frame(root)
morphological_buttons_frame.pack(side=tk.TOP, fill=tk.X)
erode_button = tk.Button(morphological_buttons_frame, text="Erode", command=lambda: apply_morphological_operation("Erode"))
dilate_button = tk.Button(morphological_buttons_frame, text="Dilate", command=lambda: apply_morphological_operation("Dilate"))
open_button = tk.Button(morphological_buttons_frame, text="Open", command=lambda: apply_morphological_operation("Open"))
close_button = tk.Button(morphological_buttons_frame, text="Close", command=lambda: apply_morphological_operation("Close"))
erode_button.pack(side=tk.LEFT, padx=10)
dilate_button.pack(side=tk.LEFT, padx=10)
open_button.pack(side=tk.LEFT, padx=10)
close_button.pack(side=tk.LEFT, padx=10)

contour_button = tk.Button(root, text="Detect Contours", command=detect_and_display_contours)
contour_button.pack()

new_functionalities_frame = tk.Frame(root)
new_functionalities_frame.pack(side=tk.TOP, fill=tk.X)
color_segmentation_button = tk.Button(new_functionalities_frame, text="Color-Based Segmentation", command=apply_color_segmentation)
detect_lines_button = tk.Button(new_functionalities_frame, text="Detect Lines", command=detect_lines)
connected_components_button = tk.Button(new_functionalities_frame, text="Connected Component Analysis", command=connected_component_analysis)

color_segmentation_button.pack(side=tk.LEFT, padx=10)
detect_lines_button.pack(side=tk.LEFT, padx=10)
connected_components_button.pack(side=tk.LEFT, padx=10)

canvas = tk.Canvas(root, width=800, height=600)
canvas.pack()

root.mainloop()
