import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def plot_luminance_distribution(image_path):
    # Load the image
    image = Image.open(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")

    # Convert the image to grayscale to get luminance
    gray_image = image.convert("L")

    # Convert the grayscale image to a numpy array
    luminance_values = np.array(gray_image).flatten()

    # Plot the distribution of luminance values
    plt.figure(figsize=(10, 6))
    plt.hist(luminance_values, bins=256, range=(0, 256), color='gray', alpha=0.75)
    plt.title(f'Distribution of Luminance Values for {image_path.split("/")[-1]}')
    plt.xlabel('Luminance')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


rootdir = "data/0_photos"
all_files = sorted([f for f in os.listdir(rootdir) if f.endswith(".jpg")], reverse=True)
for p in all_files:
    plot_luminance_distribution(os.path.join(rootdir, p))