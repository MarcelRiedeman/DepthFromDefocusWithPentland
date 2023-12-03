import cv2
import numpy as np

# Known camera intrinsic values
F=55 #mm
v0=59.06 #mm (Focus Distance = 80cm)
f1=5.6

def convert_to_grayscale(sharp_image,blurred_image):
    gray_sharp = cv2.cvtColor(sharp_image, cv2.COLOR_BGR2GRAY)
    gray_blurred = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    return(gray_sharp,gray_blurred)

def DFT_Shifted_Scaled(gray_sharp,gray_blurred):
    # Compute the discrete Fourier Transform of the images
    fourier_sharp = cv2.dft(np.float32(gray_sharp), flags=cv2.DFT_COMPLEX_OUTPUT)
    fourier_blurred = cv2.dft(np.float32(gray_blurred), flags=cv2.DFT_COMPLEX_OUTPUT)

    # Shift the zero-frequency component to the center of the spectrum
    fourier_shift_sharp = np.fft.fftshift(fourier_sharp)
    fourier_shift_blurred = np.fft.fftshift(fourier_blurred)

    # Calculate the magnitude of the Fourier Transform
    magnitude_sharp = 20 * np.log(cv2.magnitude(fourier_shift_sharp[:, :, 0], fourier_shift_sharp[:, :, 1]))
    magnitude_blurred = 20 * np.log(cv2.magnitude(fourier_shift_blurred[:, :, 0], fourier_shift_blurred[:, :, 1]))

    # Scale the magnitude for display
    scaled_magnitude_image1 = cv2.normalize(magnitude_sharp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    scaled_magnitude_image2 = cv2.normalize(magnitude_blurred, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

    # Calculate difference between natural log of images
    ln_FT_sharp = np.log(magnitude_sharp)
    ln_FT_blurred = np.log(magnitude_blurred)
    ln_difference = ln_FT_sharp - ln_FT_blurred

    return(ln_difference,scaled_magnitude_image1,scaled_magnitude_image2)

def Calculate_Depth(ln_difference,selected_indices):
    depth_sum=0
    for i in range(len(selected_indices)):
        u_array=selected_indices[i][0]
        v_array=selected_indices[i][1]
        u=np.abs(u_array-82)/164
        v=np.abs(v_array-91)/182
        bottom = 2*np.pi**2*((u**2)+(v**2))
        sigma1 = np.sqrt((ln_difference[u_array][v_array]) / (bottom))
        depth = F * v0 / (v0 - F - sigma1 * f1)
        depth_sum+=depth
    Average_Depth = depth_sum/len(selected_indices)
    return(Average_Depth)

def Calculate_Average(ln_difference,n):
    sum = 0

    for x in range(n):
        sum += np.max(ln_difference[x])
        sum += np.max(ln_difference[:, x])

    average = sum / (2 * n)
    return(average)

def Select_Indices(ln_difference,average,n):
    selected_indices = []
    for y in range(n):
        ind = np.argpartition(ln_difference[y], -5)[-5:]
        ind2 = np.argpartition(ln_difference[:, y], -5)[-5:]
        for i in range(5):
            if np.abs(ln_difference[y][ind][i] - average) < 0.2:
                # print("Suitable LN Difference:",ln_difference[y][ind][i])
                tuple = (y, ind[i])
                selected_indices.append(tuple)
                # print("Tuple to be added to selected_indices:", tuple)
            if np.abs(ln_difference[:, y][ind2][i] - average) < 0.2:
                tuple = (ind2[i], y)
                selected_indices.append(tuple)
    return selected_indices
