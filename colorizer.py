import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the blur kernel
kernel = {
    "blur" : np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
], dtype=np.float32),

"edge_detection": np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ]),  # Laplacian edge detection 
"sharpen": np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]),  # Sharpening
"inverse": np.array([
            [-1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ]),  # Inverse
"contrast": np.array([
            [0.2989, 0.5870, 0.1140],
            [0.2989, 0.5870, 0.1140],
            [0.2989, 0.5870, 0.1140]
        ]),  # Contrast
"embossing": np.array([
            [-1, 0, 0],
            [0, 0, 0],
            [0, 0, 1]
        ]),  # Embossing

"cross_edge_detection": np.array([
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0]
        ]),  # Edge detection with central emphasis

"gaussian_blur": np.array([
            [1/16, 2/16, 1/16],
            [2/16, 4/16, 2/16],
            [1/16, 2/16, 1/16]
        ])  # Gaussian blur
        } # OpenCV requires kernels to be of type float32

# Load the image in color
image = cv2.imread('WhatsApp Image 2024-05-05 at 9.23.04 PM (2).jpeg')  # Load as color (default is BGR format)

# Convert the image to RGB format for proper display with matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Apply the blur kernel using cv2.filter2D to each color channel separately
blurred_image = cv2.filter2D(image_rgb, -1, kernel["gaussian_blur"])

# Display the original and blurred images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image_rgb)  # Display original in RGB
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Blurred Image")
plt.imshow(blurred_image)  # Display blurred image
plt.axis('off')

plt.tight_layout()
plt.show()
