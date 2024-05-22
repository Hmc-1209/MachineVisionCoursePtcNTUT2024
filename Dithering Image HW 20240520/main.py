import cv2
import numpy as np


def dithering(image, matrix):
    # Process dithering
    h, w = image.shape
    mh, mw = matrix.shape
    dithered_image = np.zeros((h, w), dtype=np.uint8)
    # Normalize matrix to 0~1
    norm_matrix = matrix / (mh * mw)

    for y in range(h):
        for x in range(w):
            # Normalize pixel to 0~1
            pixel = image[y, x] / 255.0
            # Get the corresponding value from matrix
            threshold = norm_matrix[y % mh, x % mw]
            # Compare & check whether the pixel should be black or white
            if pixel > threshold:
                dithered_image[y, x] = 255
            else:
                dithered_image[y, x] = 0

    return dithered_image


def main():
    img = cv2.imread("file/input.jpg", cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Dithering matrix with size 4x4
    dithering_matrix_4x4 = np.array([
        [0, 8, 2, 10],
        [12, 4, 14, 6],
        [3, 11, 1, 9],
        [15, 7, 13, 5]
    ])
    # Dithering matrix with size 8x8
    dithering_matrix_8x8 = np.array([
        [0, 32, 8, 40, 2, 34, 10, 42],
        [48, 16, 56, 24, 50, 18, 58, 26],
        [12, 44, 4, 36, 14, 46, 6, 38],
        [60, 28, 52, 20, 62, 30, 54, 22],
        [3, 35, 11, 43, 1, 33, 9, 41],
        [51, 19, 59, 27, 49, 17, 57, 25],
        [15, 47, 7, 39, 13, 45, 5, 37],
        [63, 31, 55, 23, 61, 29, 53, 21]
    ])

    # Show all results
    cv2.imshow("Original Image", img)
    cv2.imshow("Binary Image", binary_img)
    cv2.imshow("Dithered Image 4x4 matrix", dithering(img, dithering_matrix_4x4))
    cv2.imshow("Dithered Image 8x8 matrix", dithering(img, dithering_matrix_8x8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
