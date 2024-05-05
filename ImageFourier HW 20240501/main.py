import cv2
import numpy as np
from scipy import signal


def freq_image(image):
    return np.round(np.fft.ifftshift(image).real).astype(np.uint8)


# Input image detail
img = cv2.imread("file/input3.jpeg", cv2.IMREAD_GRAYSCALE)
img = np.pad(img, ((0, 1), (1, 0)), mode='constant', constant_values=0)
print("Input image size:", img.shape)
cv2.imshow("(a) Original Image", img)
# Kernel detail
# Lowpass (Gaussian) kernel
lowpass_kernel_size = 5
lowpass_sigma = 1
lowpass_kernel = cv2.getGaussianKernel(lowpass_kernel_size, lowpass_sigma)
lowpass_kernel = lowpass_kernel * lowpass_kernel.T  # Convert 1D kernel to 2D
print("Lowpass (Gaussian) kernel:")
print(lowpass_kernel, '\n')
# Highpass (Laplacian) kernel
highpass_kernel = np.array([[0, -1, 0],
                            [-1, 4, -1],
                            [0, -1, 0]])
print("Highpass (Laplacian) kernel:")
print(highpass_kernel, '\n')

fft_img = np.fft.fft2(img)
fft_lowpass_kernel = np.fft.fft2(np.pad(lowpass_kernel, ((img.shape[0]-lowpass_kernel_size)//2,
                                                         (img.shape[1]-lowpass_kernel_size)//2)))
fft_highpass_kernel = np.fft.fft2(np.pad(highpass_kernel, ((img.shape[0]-3)//2,
                                                           (img.shape[1]-3)//2)))

cv2.imshow("(b) Original image fourier", freq_image(fft_img))
cv2.imshow("(d) Lowpass kernel fourier", freq_image(fft_lowpass_kernel))
cv2.imshow("(d) Highpass kernel fourier", freq_image(fft_highpass_kernel))

# Convolution at frequency domain
f_lowpass_shift = (fft_img * fft_lowpass_kernel)
f_highpass_shift = (fft_img * fft_highpass_kernel)

cv2.imshow("(f) Lowpass reversed Image", freq_image(f_lowpass_shift))
cv2.imshow("(f) Highpass reversed Image", freq_image(f_highpass_shift))

# Reverse fft
img_lowpass = freq_image(np.fft.ifft2(f_lowpass_shift))
img_highpass = freq_image(np.fft.ifft2(f_highpass_shift))

# Show image results
cv2.imshow("(e) Lowpass Filtered Image", img_lowpass)
cv2.imshow("(e) Highpass Filtered Image", img_highpass)


cv2.imshow("Directly lowpass convolve:", np.clip(signal.convolve2d(img, lowpass_kernel, 'same'), 0, 255).astype(np.uint8))
cv2.imshow("Directly highpass convolve:", signal.convolve2d(img, np.array(highpass_kernel).astype(np.uint8), 'same'))

cv2.waitKey(0)
cv2.destroyAllWindows()
