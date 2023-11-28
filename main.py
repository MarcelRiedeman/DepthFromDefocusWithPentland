import math
from PIL import Image
import numpy as np
from scipy.fft import fft2, fftshift
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

image_patch1_path = "images/patch F5.6-80.png"
image_patch2_path = "images/patch F16-80.png"


def resize_image(image_path, new_size=(120, 120)):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    resized_image = image.resize(new_size, Image.ANTIALIAS)  # Use 'Image.ANTIALIAS' for older versions of Pillow
    return np.array(resized_image)


def image_to_dft(image_path, new_size=(120, 120)):
    # Resize the image to the specified size
    resized_image = resize_image(image_path, new_size)

    # Perform 2D Fourier Transform
    fourier_transform = fft2(resized_image)

    # Shift zero frequency components to the center
    fourier_transform_shifted = fftshift(fourier_transform)

    # Calculate the magnitude spectrum
    magnitude_spectrum = np.abs(fourier_transform_shifted)

    return fourier_transform, magnitude_spectrum


def depth_from_patch(DFT_patch1, DFT_patch2, D_1, D_2, f, u, v, s):
    # Ensure the logarithms are computed for non-negative values
    log_DFT_patch1 = np.log(np.maximum(DFT_patch1, 1e-10))
    log_DFT_patch2 = np.log(np.maximum(DFT_patch2, 1e-10))

    sigma_2 = math.sqrt(log_DFT_patch2 - log_DFT_patch1) / (
                2 * np.pi ** 2 * (u ** 2 + v ** 2) * ((D_1 ** 2 / D_2 ** 2) - 1))
    depth = s * f / (s - f + 2 * sigma_2 * (f / D_2))

    return depth


def plot_image_and_dft(image_path):
    # Get DFT and magnitude spectrum
    dft, magnitude_spectrum = image_to_dft(image_path)

    # Display the original image and its Fourier Transform
    plt.figure(figsize=(12, 6))

    plt.subplot(121), plt.imshow(Image.open(image_path), cmap='gray')
    plt.title('Original Image'), plt.axis('off')

    plt.subplot(122), plt.imshow(np.log(1 + magnitude_spectrum), cmap='gray', norm=LogNorm())
    plt.title('Fourier Transform'), plt.axis('off')

    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    plot_image_and_dft(image_patch1_path)
    # Depth Calculation
    new_size = (120, 120)
    dft_patch1, _ = image_to_dft(image_patch1_path, new_size)
    dft_patch2, _ = image_to_dft(image_patch2_path, new_size)

    # Parameters for depth calculation
    D_1 = 1.0  # Replace with actual values
    D_2 = 2.0  # Replace with actual values
    f = 50.0  # Replace with actual values
    u, v = 0.0, 0.0  # Replace with actual values
    s = 100.0  # Replace with actual values

    calculated_depth = depth_from_patch(dft_patch1, dft_patch2, D_1, D_2, f, u, v, s)
    print(f"Calculated depth: {calculated_depth}")
