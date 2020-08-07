import cv2
import numpy as np

img = cv2.imread('shapes.png', 0)

# smoothing the image:
cv2.GaussianBlur(img, (3, 3), 0)

cv2.imshow('original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Taking dimensions of image:
rows, cols = np.shape(img)

# Thresholding the image:
thresh_img = np.where(img < 240, 255, 0)
thresh_img = np.uint8(thresh_img)
cv2.imshow('threshold image', thresh_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Pixel value which will be used as indicator to Objects:
pix_val = 1


while True:

    # finding white pixels to assign them pix_val:
    for i in range(rows):
        for j in range(cols):
            if thresh_img[i][j] == 255:
                thresh_img[i][j] = pix_val
                break
        else:
            continue
        break  # executes when inner loop breaks

    # assigning 8 neighbourhood white pixels(255) same pixel value:
    for i in range(rows):
        for j in range(cols):
            try:
                if thresh_img[i][j] == 255:
                    if thresh_img[i+1][j] == pix_val or thresh_img[i+1][j+1] == pix_val or thresh_img[i][j+1] == pix_val or thresh_img[i-1][j+1] == pix_val or thresh_img[i-1][j] == pix_val or thresh_img[i-1][j-1] == pix_val or thresh_img[i][j-1] == pix_val or thresh_img[i+1][j-1] == pix_val:
                        thresh_img[i][j] = pix_val
                    else:
                        pass
            except IndexError:
                pass

    if 255 not in thresh_img:
        break

    pix_val = pix_val + 1


# Taking median blur of 8 neighbourhood pixels for sharp object detection:
cv2.medianBlur(thresh_img, 3)

# mapping different objects to different hue values:
screen_hue = np.uint8(180*thresh_img/np.max(thresh_img))

# setting saturation and value/intensity to max i.e. 255:
blank_ch = 255*np.ones_like(screen_hue)

# merging hue, sat and val to make HSV image:
labeled_img = cv2.merge([screen_hue, blank_ch, blank_ch])

# converting to BGR for display
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

# After converting image HSV2BGR setting background to black for bgr image:
labeled_img[screen_hue == 0] = 0
cv2.imshow('labeled_img', labeled_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
