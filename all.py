1 – MORPHOLOGY
import cv2
import numpy as np
from skimage.morphology import skeletonize, thin
import matplotlib.pyplot as plt

# --------------------------------------------------
# Padding Helper
# --------------------------------------------------
def pad_image(img, kernel):
    pad_h = kernel.shape[0] // 2
    pad_w = kernel.shape[1] // 2
    return np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)

# ==================================================
# ========  BINARY MORPHOLOGY IMPLEMENTATIONS  ======
# ==================================================
def erosion_binary(img, kernel):
    img_padded = pad_image(img, kernel)
    output = np.zeros_like(img)
    kh, kw = kernel.shape
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            region = img_padded[i:i+kh, j:j+kw]
            if np.all(region[kernel == 1] == 255):
                output[i, j] = 255
    return output

def dilation_binary(img, kernel):
    img_padded = pad_image(img, kernel)
    output = np.zeros_like(img)
    kh, kw = kernel.shape
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            region = img_padded[i:i+kh, j:j+kw]
            if np.any(region[kernel == 1] == 255):
                output[i, j] = 255
    return output

def opening_binary(img, kernel):
    return dilation_binary(erosion_binary(img, kernel), kernel)

def closing_binary(img, kernel):
    return erosion_binary(dilation_binary(img, kernel), kernel)

def gradient_binary(img, kernel):
    return dilation_binary(img, kernel) - erosion_binary(img, kernel)

def top_hat_white_binary(img, kernel):
    opened = opening_binary(img, kernel)
    return cv2.subtract(img, opened)

def top_hat_black_binary(img, kernel):
    closed = closing_binary(img, kernel)
    return cv2.subtract(closed, img)

def hit_or_miss(img, kernel):
    img_bin = (img // 255).astype(np.uint8)
    k1 = np.uint8(kernel == 1)
    k0 = np.uint8(kernel == 0)
    print(k1,"k1\n\n\n")
    eroded1 = erosion_binary(img, k1)
    eroded0 = erosion_binary(255 - img, k0)
    return np.minimum(eroded1, eroded0)

def skeletonization(img):
    binary = (img // 255).astype(bool)
    skeleton = skeletonize(binary)
    return (skeleton * 255).astype(np.uint8)

def thinning(img):
    binary = (img // 255).astype(bool)
    thin_img = thin(binary)
    return (thin_img * 255).astype(np.uint8)

def pruning(skeleton_img):
    kernel = np.ones((3, 3), np.uint8)
    pruned = erosion_binary(skeleton_img, kernel)
    return pruned

# ==================================================
# ========  GRAY-LEVEL MORPHOLOGY IMPLEMENTS  ======
# ==================================================
def erosion_gray(img, kernel):
    img_padded = pad_image(img, kernel)
    output = np.zeros_like(img)
    kh, kw = kernel.shape
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            region = img_padded[i:i+kh, j:j+kw]
            output[i, j] = np.min(region[kernel == 1])
    return output

def dilation_gray(img, kernel):
    img_padded = pad_image(img, kernel)
    output = np.zeros_like(img)
    kh, kw = kernel.shape
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            region = img_padded[i:i+kh, j:j+kw]
            output[i, j] = np.max(region[kernel == 1])
    return output

def opening_gray(img, kernel):
    return dilation_gray(erosion_gray(img, kernel), kernel)

def closing_gray(img, kernel):
    return erosion_gray(dilation_gray(img, kernel), kernel)

def gradient_gray(img, kernel):
    return dilation_gray(img, kernel) - erosion_gray(img, kernel)

def top_hat_white_gray(img, kernel):
    opened = opening_gray(img, kernel)
    return cv2.subtract(img, opened)

def top_hat_black_gray(img, kernel):
    closed = closing_gray(img, kernel)
    return cv2.subtract(closed, img)

# ==================================================
# ================== MAIN EXECUTION =================
# ==================================================
if __name__ == "__main__":
    # Load grayscale image
    img = cv2.imread('/content/Screenshot 2023-12-08 211529.png', cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image not found! Please check the path.")

    # Binary conversion
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Structuring element
    kernel = np.array([[0,1,0],
                       [1,1,1],
                       [0,1,0]], dtype=np.uint8)

    # ======== Binary Morphology ========
    ero_b = erosion_binary(binary, kernel)
    dil_b = dilation_binary(binary, kernel)
    opn_b = opening_binary(binary, kernel)
    cls_b = closing_binary(binary, kernel)
    grad_b = gradient_binary(binary, kernel)
    topw_b = top_hat_white_binary(binary, kernel)
    topb_b = top_hat_black_binary(binary, kernel)
    hm = hit_or_miss(binary, kernel)
    skel = skeletonization(binary)
    thin_img = thinning(binary)
    pruned = pruning(skel)

    # ======== Gray-Level Morphology ========
    ero_g = erosion_gray(img, kernel)
    dil_g = dilation_gray(img, kernel)
    opn_g = opening_gray(img, kernel)
    cls_g = closing_gray(img, kernel)
    grad_g = gradient_gray(img, kernel)
    topw_g = top_hat_white_gray(img, kernel)
    topb_g = top_hat_black_gray(img, kernel)

    # ======== Display All ========
    titles = [
        'Binary Erosion', 'Binary Dilation', 'Binary Opening', 'Binary Closing',
        'Binary Gradient', 'Top Hat (White)', 'Top Hat (Black)', 'Hit-or-Miss',
        'Skeletonization', 'Thinning', 'Pruning',
        'Gray Erosion', 'Gray Dilation', 'Gray Opening', 'Gray Closing',
        'Gray Gradient', 'Gray Top Hat (White)', 'Gray Top Hat (Black)'
    ]
    images = [
        ero_b, dil_b, opn_b, cls_b,
        grad_b, topw_b, topb_b, hm,
        skel, thin_img, pruned,
        ero_g, dil_g, opn_g, cls_g,
        grad_g, topw_g, topb_g
    ]

    plt.figure(figsize=(18, 18))
    for i in range(len(images)):
        plt.subplot(5, 4, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()



2 – FEATURE EXTRACTION
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import blob_log
import importlib

# --- Handle scikit-image GLCM imports for both old and new versions ---
try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    from skimage.feature import greycomatrix as graycomatrix
    from skimage.feature import greycoprops as graycoprops

# ========== 1. Read Image in Grayscale ==========
img = cv2.imread('cam.jpeg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found. Please check the file path.")

plt.figure(figsize=(6, 6))
plt.imshow(img, cmap='gray')
plt.title("Original Grayscale Image")
plt.axis('off')
plt.show()

# ========== 2. Corner Detection ==========
# Harris Corner Detection
harris = cv2.cornerHarris(np.float32(img), 2, 3, 0.04)
harris = cv2.dilate(harris, None)
img_harris = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
img_harris[harris > 0.01 * harris.max()] = [0, 0, 255]

# Shi-Tomasi Corner Detection
corners = cv2.goodFeaturesToTrack(img, 100, 0.01, 10)
img_shi = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
if corners is not None:
    for c in corners:
        x, y = c.ravel()
        cv2.circle(img_shi, (int(x), int(y)), 3, (0, 255, 0), -1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(img_harris, cv2.COLOR_BGR2RGB))
plt.title("Harris Corners")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(img_shi, cv2.COLOR_BGR2RGB))
plt.title("Shi-Tomasi Corners")
plt.axis('off')
plt.show()

# ========== 3. Blob Detection (Laplacian of Gaussian) ==========
blobs = blob_log(img, max_sigma=30, num_sigma=10, threshold=0.05)
img_blob = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for blob in blobs:
    y, x, r = blob
    cv2.circle(img_blob, (int(x), int(y)), int(r * np.sqrt(2)), (255, 0, 0), 2)

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(img_blob, cv2.COLOR_BGR2RGB))
plt.title("Blob Detection (LoG)")
plt.axis('off')
plt.show()

# ========== 4. Texture Analysis (GLCM) ==========
glcm = graycomatrix(img, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                    levels=256, symmetric=True, normed=True)

contrast = graycoprops(glcm, 'contrast')
homogeneity = graycoprops(glcm, 'homogeneity')
energy = graycoprops(glcm, 'energy')
correlation = graycoprops(glcm, 'correlation')

print("=== Texture Analysis (GLCM Features) ===")
print(f"Contrast: {contrast.mean():.4f}")
print(f"Homogeneity: {homogeneity.mean():.4f}")
print(f"Energy: {energy.mean():.4f}")
print(f"Correlation: {correlation.mean():.4f}")

# ========== 5. Feature Descriptors (SIFT, SURF, ORB) ==========

# --- SIFT ---
sift = cv2.SIFT_create()
kp_sift, des_sift = sift.detectAndCompute(img, None)
img_sift = cv2.drawKeypoints(img, kp_sift, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)


# --- ORB ---
orb = cv2.ORB_create()
kp_orb, des_orb = orb.detectAndCompute(img, None)
img_orb = cv2.drawKeypoints(img, kp_orb, None,
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# --- Display SIFT, SURF, ORB ---
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img_sift, cv2.COLOR_BGR2RGB))
plt.title("SIFT Features")
plt.axis('off')


plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB))
plt.title("ORB Features")
plt.axis('off')
plt.show()


3 - TRANSFORMS
import cv2, pywt
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("image path", cv2.IMREAD_GRAYSCALE)
f32 = np.float32(img)

# ---------- DCT ----------
dct = cv2.dct(f32)
idct = cv2.idct(dct)

# ---------- DWT / Haar ----------
cA, (cH, cV, cD) = pywt.dwt2(f32, 'haar')
idwt = pywt.idwt2((cA, (cH, cV, cD)), 'haar')


plt.figure(figsize=(12,6))
plt.subplot(231); plt.imshow(img, cmap='gray'); plt.title("Original"); plt.axis('off')
plt.subplot(232); plt.imshow(np.log1p(abs(dct)), cmap='gray'); plt.title("DCT Magnitude"); plt.axis('off')
plt.subplot(233); plt.imshow(np.uint8(idct), cmap='gray'); plt.title("IDCT"); plt.axis('off')
plt.subplot(234); plt.imshow(np.vstack([np.hstack([cA, cH]), np.hstack([cV, cD])]), cmap='gray'); plt.title("Haar DWT"); plt.axis('off')
plt.subplot(235); plt.imshow(np.uint8(idwt), cmap='gray'); plt.title("IDWT"); plt.axis('off')

plt.tight_layout();

4-

def skeletonization_custom(img, kernel=None):
    if kernel is None:
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))

    img = (img > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(img)
    temp = np.zeros_like(img)

    while True:
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        temp = cv2.subtract(img, opened)
        skeleton = cv2.bitwise_or(skeleton, temp)
        img = eroded.copy()
        if cv2.countNonZero(img) == 0:
            break

    return skeleton

5-

# ========================= TRANSFORM ========================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale image
img = cv2.imread('/content/sample.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Image not found! Please check the path.")

# ========== 1. FOURIER TRANSFORM ==========
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

# Inverse FFT
f_ishift = np.fft.ifftshift(fshift)
img_reconstructed_fft = np.abs(np.fft.ifft2(f_ishift))

# ========== 2. DISCRETE COSINE TRANSFORM (DCT) ==========
dct = cv2.dct(np.float32(img))
dct_log = np.log(np.abs(dct) + 1)
img_reconstructed_dct = cv2.idct(dct)

# ========== 3. MANUAL HAAR WAVELET TRANSFORM ==========
def haar_wavelet_transform(image):
    img = np.float32(image)
    h, w = img.shape
    low_rows = (img[:, 0::2] + img[:, 1::2]) / 2
    high_rows = (img[:, 0::2] - img[:, 1::2]) / 2
    cA = (low_rows[0::2, :] + low_rows[1::2, :]) / 2
    cH = (high_rows[0::2, :] + high_rows[1::2, :]) / 2
    cV = (low_rows[0::2, :] - low_rows[1::2, :]) / 2
    cD = (high_rows[0::2, :] - high_rows[1::2, :]) / 2
    return cA, cH, cV, cD

cA, cH, cV, cD = haar_wavelet_transform(img)

def inverse_haar_wavelet(cA, cH, cV, cD):
    reconstructed = np.zeros((cA.shape[0]*2, cA.shape[1]*2), dtype=np.float32)
    reconstructed[0::2, 0::2] = cA + cH + cV + cD
    reconstructed[0::2, 1::2] = cA - cH + cV - cD
    reconstructed[1::2, 0::2] = cA + cH - cV - cD
    reconstructed[1::2, 1::2] = cA - cH - cV + cD
    return reconstructed / 2

img_reconstructed_dwt = inverse_haar_wavelet(cA, cH, cV, cD)

# ========== 4. HOUGH TRANSFORM ==========
edges = cv2.Canny(img, 100, 200)

# --- Hough Lines ---
lines_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
if lines is not None:
    for rho, theta in lines[:, 0]:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + 1000 * (-b)), int(y0 + 1000 * (a))
        x2, y2 = int(x0 - 1000 * (-b)), int(y0 - 1000 * (a))
        cv2.line(lines_img, (x1, y1), (x2, y2), (0, 0, 255), 1)

# --- Hough Circles ---
circles_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 30,
                           param1=100, param2=30, minRadius=10, maxRadius=100)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(circles_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.circle(circles_img, (i[0], i[1]), 2, (0, 0, 255), 3)

# ========== DISPLAY RESULTS ==========
plt.figure(figsize=(14, 12))

plt.subplot(4, 3, 1), plt.imshow(img, cmap='gray'), plt.title('Original')
plt.subplot(4, 3, 2), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('FFT Spectrum')
plt.subplot(4, 3, 3), plt.imshow(img_reconstructed_fft, cmap='gray'), plt.title('Inverse FFT')

plt.subplot(4, 3, 4), plt.imshow(dct_log, cmap='gray'), plt.title('DCT (log scale)')
plt.subplot(4, 3, 5), plt.imshow(img_reconstructed_dct, cmap='gray'), plt.title('Inverse DCT')

plt.subplot(4, 3, 6), plt.imshow(cA, cmap='gray'), plt.title('DWT Approximation (cA)')
plt.subplot(4, 3, 7), plt.imshow(cH, cmap='gray'), plt.title('DWT Horizontal (cH)')
plt.subplot(4, 3, 8), plt.imshow(cV, cmap='gray'), plt.title('DWT Vertical (cV)')
plt.subplot(4, 3, 9), plt.imshow(cD, cmap='gray'), plt.title('DWT Diagonal (cD)')

plt.subplot(4, 3, 10), plt.imshow(edges, cmap='gray'), plt.title('Canny Edges')
plt.subplot(4, 3, 11), plt.imshow(cv2.cvtColor(lines_img, cv2.COLOR_BGR2RGB)), plt.title('Hough Lines')
plt.subplot(4, 3, 12), plt.imshow(cv2.cvtColor(circles_img, cv2.COLOR_BGR2RGB)), plt.title('Hough Circles')

plt.tight_layout()
plt.show()

