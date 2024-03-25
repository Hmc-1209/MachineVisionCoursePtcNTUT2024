import numpy as np
import cv2
from matplotlib import pyplot as plt


def load_image(file_path):
    # Loading image & checking information
    img = cv2.imread(file_path, 0)
    height = img.shape[0]
    width = img.shape[1]
    print(f"---------------------\nOriginal image info:"
          f"\nImage height:{height}\n"
          f"Image width:{width}\n---------------------")
    return img, height, width


def image_preprocess(img, width, height):
    img = cv2.resize(img, (width, height))
    img = cv2.blur(img, (5, 5))
    return img


def split_image(img, width):
    # Splitting original image horizontally
    middle = width // 2
    left = img[:, 0:middle]
    right = img[:, middle:width]
    return left, right


def merge_and_process_image(left_img, right_img):
    # Merging left & right image
    stereo = cv2.StereoBM.create(numDisparities=48, blockSize=15)
    disparity = stereo.compute(left_img, right_img)
    disparity = cv2.dilate(disparity, (15, 15))
    disparity = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return disparity


def show_images(img, final_img):
    # Display original image and processed image
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axs[0].axis('off')
    axs[1].imshow(final_img)
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()


def main():
    # Get original image
    origin_img, height, width = load_image('./file/input.png')
    # Image preprocess
    img = image_preprocess(origin_img, int(width / 6), int(height / 6))
    # Split the image
    origin_left_img, origin_right_img = split_image(origin_img, width)
    left_img, right_img = split_image(img, int(width / 6))
    # Merge and process images
    disparity = merge_and_process_image(left_img, right_img)
    disparity = cv2.resize(disparity, (int(width / 2), height))
    # Display both original and processed images
    cv2.imshow("Disparity", disparity)
    depth_mask = np.where(disparity > 90, 255, 0).astype(np.uint8)
    masked_gray_image = cv2.bitwise_and(origin_right_img, origin_right_img, mask=depth_mask)
    cv2.imshow("final", masked_gray_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
