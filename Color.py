import cv2
import numpy as np

def lab_color_correction(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    A_mean = np.mean(A)
    B_mean = np.mean(B)
    A = np.clip(A - (A_mean - 128), 0, 255).astype(np.uint8)
    B = np.clip(B - (B_mean - 128), 0, 255).astype(np.uint8)
    corrected_lab = cv2.merge([L, A, B])
    corrected_image = cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2RGB)
    return corrected_image

# 基于Lab空间的直方图拉伸
def lab_histogram_stretching(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    L_min, L_max = np.percentile(L, [1, 99])
    L = np.clip((L - L_min) / (L_max - L_min) * 255, 0, 255).astype(np.uint8)
    stretched_lab = cv2.merge([L, A, B])
    stretched_image = cv2.cvtColor(stretched_lab, cv2.COLOR_LAB2RGB)
    return stretched_image


def white_balance_adaptive(img):
    """Perform adaptive white balance and color balance on an image."""
    # 颜色校正
    corrected_img = lab_color_correction(img)

    # 直方图拉伸
    balanced_img = lab_histogram_stretching(corrected_img)

    return balanced_img