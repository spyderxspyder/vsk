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
