import math
import numpy as np
from scipy.fft import fft2, fftshift
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

image_patch1_path = "images/patch F5.6-80.png"
image_patch2_path = "images/patch F16-80.png"


def resize_image(image_path, new_size=(120, 120)):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    resized_image = image.resize(new_size, resample=Image.LANCZOS)
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


def find_proper_u_v(magnitude_patch1, magnitude_patch2):
    u, v = 100, 100
    while u < 120:
        while v < 120:
            # Your logic here using DFT_patch1[u, v] and DFT_patch2[u, v]
            if np.log(magnitude_patch2[u, v]) > np.log(magnitude_patch1[u, v]):
                # Your code for the condition when DFT_patch1 and DFT_patch2 satisfy the condition
                print(u, v)
                print(magnitude_patch1[u, v], magnitude_patch2[u, v])
                return u, v
            v += 1
        u += 1
        v = 0  # Reset v for the next iteration of the outer loop


def depth_from_patch(magnitude_patch1, magnitude_patch2, F_1, F_2, f, s):
    u, v = find_proper_u_v(magnitude_patch1, magnitude_patch2)

    # Ensure the logarithms are computed for non-negative values
    log_mag_patch1 = np.log(magnitude_patch1[u, v])
    log_mag_patch2 = np.log(magnitude_patch2[u, v])

    sigma_2 = math.sqrt(log_mag_patch2 - log_mag_patch1) / (
            2 * np.pi ** 2 * (u ** 2 + v ** 2) * (((F_2 / F_1) ** 2) - 1))
    print("sigma_2: " + str(sigma_2))
    depth = s * f / (s - f + 2 * sigma_2 * (F_1))

    return depth


def plot_image_and_dft(image_path, new_size=(120, 120)):
    # Get DFT and magnitude spectrum
    dft, magnitude_spectrum = image_to_dft(image_path, new_size)

    # Display the original image and its Fourier Transform
    plt.figure(figsize=(12, 6))

    plt.subplot(121), plt.imshow(resize_image(image_path, new_size), cmap='gray')
    plt.title('Original Image'), plt.axis('off')

    plt.subplot(122), plt.imshow(np.log(1 + magnitude_spectrum), cmap='gray', norm=LogNorm())
    plt.title('Fourier Transform'), plt.axis('off')

    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    plot_image_and_dft(image_patch1_path)
    # Example usage
    new_size = (120, 120)
    dft_patch1, mag_patch1 = image_to_dft(image_patch1_path, new_size)
    dft_patch2, mag_patch2 = image_to_dft(image_patch2_path, new_size)

    # Parameters for depth calculation
    F_1 = 5.6
    F_2 = 16
    f = 55.0  # mm
    s = 61.8  # mm

    calculated_depth = depth_from_patch(mag_patch1, mag_patch2, F_1, F_2, f, s)
    print(f"Calculated depth: {calculated_depth}")
