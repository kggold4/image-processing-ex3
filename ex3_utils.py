from typing import List
import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt

MY_ID = 208980359
MIN_MSE = 1000
STEPS = 100
FULL_DEGREES = 360
WIN_13 = 13
WIN_5 = 5
BASE_KERNEL = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])


def myID() -> np.int:
    """
    Return my ID
    :return: int
    """
    return MY_ID


def get_base_kernel(x, y) -> np.array:
    return np.array([[1, 0, x], [0, 1, y], [0, 0, 1]], dtype=np.float)


def get_sigma(kernel_size, is_round: bool = False):
    result = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
    if is_round:
        result = int(round(result))
    return result


def get_list_t() -> list:
    return [(np.array([0]), 0, 0)]


def double_win(win):
    return win * win


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> (np.ndarray, np.ndarray):
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    w = win_size // 2
    X = cv2.filter2D(im2, -1, BASE_KERNEL, borderType=cv2.BORDER_REPLICATE)
    Y = cv2.filter2D(im2, -1, BASE_KERNEL.transpose(), borderType=cv2.BORDER_REPLICATE)
    Z = im2 - im1

    UV, JI = [], []
    t = 0
    for i in range(step_size, im1.shape[0], step_size):
        for j in range(step_size, im1.shape[1], step_size):

            NX = X[i - w:1 + i + w, j - w:1 + j + w]
            NY = Y[i - w:1 + i + w, j - w:1 + j + w]
            NZ = Z[i - w:1 + i + w, j - w:1 + j + w]
            NX = NX.flatten()
            NY = NY.flatten()
            NZ = NZ.flatten()

            LEN_NX = len(NX)
            LEN_NY = len(NY)

            A = np.array([[sum(NX[t] ** 2 for t in range(LEN_NX)), sum(NY[t] * NX[t] for t in range(LEN_NX))],
                          [sum(NX[t] * NY[t] for t in range(LEN_NX)), sum(NY[t] ** 2 for t in range(LEN_NY))]])

            B = np.array([[-1 * sum(NX[t] * NZ[t] for t in range(LEN_NX)),
                           -1 * sum(NY[t] * NZ[t] for t in range(LEN_NY))]]).reshape(2, 1)

            ev1, ev2 = np.linalg.eigvals(A)
            if ev2 < ev1:
                ev1, ev2 = ev2, ev1

            if ev2 >= ev1 > 1 and ev2 / ev1 < STEPS:
                velo = np.dot(np.linalg.pinv(A), B)
                u, v = velo[0][0], velo[1][0]
                UV.append(np.array([u, v]))
            else:
                t += 1
                UV.append(np.array([0.0, 0.0]))

            JI.append(np.array([j, i]))
    return np.array(JI), np.array(UV)


def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int, stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    p_d = []
    if img1.ndim > 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    UVS = np.zeros((*img2.shape, 2))
    for i in range(k):
        p_d.append(np.array([img1.copy(), img2.copy()]))

        img1 = cv2.pyrDown(img1, dstsize=(img1.shape[1] // 2, img1.shape[0] // 2))
        img2 = cv2.pyrDown(img2, dstsize=(img2.shape[1] // 2, img2.shape[0] // 2))
        if img1.ndim < 2:
            k = i
            break
    for level in range(k - 1, -1, -1):
        pyr_1, pyr_2 = p_d[level]
        dx_median, dy_median = np.ma.median(np.ma.masked_where(UVS == np.zeros(2), UVS), axis=(0, 1)).filled(0)
        pyr_1 = cv2.warpPerspective(pyr_1, get_base_kernel(x=dx_median, y=dy_median), pyr_1.shape[::-1])
        PTs, uv = opticalFlow(pyr_1, pyr_2, max(int(stepSize * (2 ** -level)), 1), winSize)
        if PTs.size == 0:
            continue
        converted_points = (2 ** level) * PTs
        UVS[converted_points[:, 1], converted_points[:, 0]] += uv * 2

    return UVS


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    diff = np.inf
    spot = 0
    _, im_new = opticalFlow(im1, im2, 10, 5)

    # look at all the u,v we found
    for k in range(len(im_new)):
        t1, t2 = im_new[k][0], im_new[k][1]
        t = get_base_kernel(x=t1, y=t2)

        # create a new image
        new_img = cv2.warpPerspective(im1, t, (im1.shape[1], im1.shape[0]))

        # find the smallest diff
        d = ((im2 - new_img) ** 2).sum()

        if d < diff:
            diff, spot = d, k
            if diff == 0:
                break
    t1, t2 = im_new[spot][0], im_new[spot][1]
    return get_base_kernel(x=t1, y=t2)


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    t = 0
    min_mse = MIN_MSE
    for t in range(FULL_DEGREES):
        tmp_t = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]], dtype=np.float64)
        img_by_t = cv2.warpPerspective(im1, tmp_t, im1.shape[::-1])
        mse = np.square(np.subtract(im2, img_by_t)).mean()
        if mse < min_mse:
            min_mse = mse
            M = tmp_t
    rigid_mat = np.array([[np.cos(t), np.sin(t), 0], [-np.sin(t), np.cos(t), 0], [0, 0, 1]], dtype=np.float64)
    M = findTranslationLK(im1, cv2.warpPerspective(im2, rigid_mat, im2.shape[::-1]))
    ans = np.array(
        [[np.cos(t), -np.sin(t), M[0, 2], ], [np.sin(t), np.cos(t), M[1, 2]], [0, 0, 1]], dtype=np.float64)
    return ans


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    win = WIN_13
    win_pad = win // 2
    image_to_pad = cv2.copyMakeBorder(im2, win_pad, win_pad, win_pad, win_pad, cv2.BORDER_REPLICATE, None, value=0)
    I, J = [], []
    for x in range(1, 5):
        I.append(x * (im1.shape[0] // 5))
        J.append(x * (im1.shape[1] // 5))

    list_t = get_list_t()
    for x in range(len(I)):
        for y in range(len(J)):
            window_a = im1[I[x] - win_pad:I[x] + win_pad + 1, J[y] - win_pad:J[y] + win_pad + 1]
            a = window_a.reshape(1, double_win(win))
            a_transpose = a.T
            big_list_t = get_list_t()
            for i in range(0, im2.shape[0]):
                for j in range(0, im2.shape[1]):
                    if (i + win_pad + win) < image_to_pad.shape[0] and (j + win_pad + win) < image_to_pad.shape[1]:
                        window_b = image_to_pad[i + win_pad:i + win_pad + win, j + win_pad:j + win_pad + win]
                        b = window_b.reshape(1, double_win(win))
                        b_transpose = b.T
                        top = np.dot(a, b_transpose)
                        bottom = np.dot(a, a_transpose) + np.dot(b, b_transpose)
                        if bottom != 0:
                            cor = top / bottom
                            if cor > big_list_t[0][0]:
                                big_list_t.clear()
                                big_list_t.insert(0, (cor, i, j))
                            elif cor == big_list_t[0][0]:
                                big_list_t.insert(0, (cor, i, j))
            if big_list_t[0][0][0] > list_t[0][0][0]:
                list_t.clear()
                for k in range(len(big_list_t)):
                    list_t.append((big_list_t[k], (I[x], J[y])))

            if big_list_t[0][0][0] == list_t[0][0][0]:
                for k in range(len(big_list_t)):
                    list_t.append((big_list_t[k], (I[x], J[y])))

    dif = np.inf
    spot = -1
    for x in range(len(list_t)):
        t1, t2 = list_t[x][1][0] - list_t[x][0][1], list_t[x][1][1] - list_t[x][0][2]
        new = cv2.warpPerspective(im1, get_base_kernel(x=t1, y=t2), (im1.shape[1], im1.shape[0]))
        d = ((im2 - new) ** 2).sum()
        if d < dif:
            dif = d
            spot = x
            if dif == 0:
                break

    return get_base_kernel(x=list_t[spot][1][0] - list_t[spot][0][1], y=list_t[spot][1][1] - list_t[spot][0][2])


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    win = WIN_5
    pad = win // 2
    im2pad = cv2.copyMakeBorder(im2, pad, pad, pad, pad, cv2.BORDER_REPLICATE, None, value=0)
    I = []
    J = []
    for x in range(1, WIN_5):
        I.append((im1.shape[0] // WIN_5) * x)
        J.append((im1.shape[1] // WIN_5) * x)

    list_t = get_list_t()
    for x in range(len(I)):
        for y in range(len(J)):
            window_a = im1[I[x] - pad:I[x] + pad + 1, J[y] - pad:J[y] + pad + 1]
            a = window_a.reshape(1, double_win(win))
            aT = a.T
            big_list_t = get_list_t()
            for i in range(0, im2.shape[0]):
                for j in range(0, im2.shape[1]):
                    if (i + pad + win) < im2pad.shape[0] and (j + pad + win) < im2pad.shape[1]:

                        window_b = im2pad[i + pad:i + pad + win, j + pad:j + pad + win]

                        b = window_b.reshape(1, double_win(win))
                        bT = b.T
                        top = np.dot(a, bT)
                        bottom = np.dot(a, aT) + np.dot(b, bT)

                        if bottom != 0:
                            cor = top / bottom
                            if cor > big_list_t[0][0]:
                                big_list_t.clear()
                                big_list_t.insert(0, (cor, i, j))
                            elif cor == big_list_t[0][0]:
                                big_list_t.insert(0, (cor, i, j))

            if big_list_t[0][0][0] > list_t[0][0][0]:
                list_t.clear()
                for m in range(len(big_list_t)):
                    list_t.append((big_list_t[m], (I[x], J[y])))
            if big_list_t[0][0][0] == list_t[0][0][0]:
                for m in range(len(big_list_t)):
                    list_t.append((big_list_t[m], (I[x], J[y])))

    spot = -1
    diff = np.inf
    for n in range(len(list_t)):
        x = list_t[n][1][0] - list_t[n][0][1]
        y = list_t[n][1][1] - list_t[n][0][2]

        if y != 0:
            t = np.arctan(x / y)
        else:
            t = 0
        t = np.array([[np.cos(t), -np.sin(t), x], [np.sin(t), np.cos(t), y], [0, 0, 1]], dtype=np.float)
        new = cv2.warpPerspective(im1, t, im1.shape[::-1])
        d = ((im2 - new) ** 2).sum()
        if d < diff:
            diff = d
            spot = n
        if diff == 0:
            break
    x = list_t[spot][1][0] - list_t[spot][0][1]
    y = list_t[spot][1][1] - list_t[spot][0][2]

    t = 0
    if y != 0:
        t += np.arctan(x / y)

    return np.array([[np.cos(t), -np.sin(t), x], [np.sin(t), np.cos(t), y], [0, 0, 1]], dtype=np.float)


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    result = np.zeros((im1.shape[0], im1.shape[1]))
    t_inv = np.linalg.inv(T)
    print(T, "\n", t_inv)
    for i in range(im2.shape[0]):
        for j in range(im2.shape[1]):
            arr = np.array([i, j, 1])
            t = t_inv.dot(arr)

            x1 = np.floor(t[0]).astype(int)
            x2 = np.ceil(t[0]).astype(int)
            x3 = np.round(t[0] % 1, 3)

            y1 = np.floor(t[1]).astype(int)
            y2 = np.ceil(t[1]).astype(int)
            y3 = np.round(t[1] % 1, 3)

            if 0 <= x1 < im1.shape[0] and 0 <= y1 < im1.shape[1]:
                result[i][j] += im1[x1][y1] * (1 - x3) * (1 - y3)

            if 0 <= x2 < im1.shape[0] and 0 <= y1 < im1.shape[1]:
                result[i][j] += im1[x2][y1] * x3 * (1 - y3)

            if 0 <= x1 < im1.shape[0] and 0 <= y2 < im1.shape[1]:
                result[i][j] += im1[x1][y2] * (1 - x3) * y3

            if 0 <= x2 < im1.shape[0] and 0 <= y2 < im1.shape[1]:
                result[i][j] += im1[x2][y2] * x3 * y3

    plt.imshow(result)
    plt.show()
    return result


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    img_x_shape = img.shape[0]
    img_y_shape = img.shape[1]
    img = img[0: 2 ** levels * int(img_x_shape / 2 ** levels), 0: 2 ** levels * int(img_y_shape / 2 ** levels)]
    temp_img = img.copy()
    pyr = [temp_img]
    for i in range(levels - 1):
        temp_img = blurImage2(temp_img, 5)
        temp_img = temp_img[::2, ::2]
        pyr.append(temp_img)
    return pyr


def blurImage2(in_image: np.ndarray, kernel_size: int) -> np.ndarray:
    """
    Blur an image using a Gaussian kernel using OpenCV built-in functions
    :param in_image: Input image
    :param kernel_size: Kernel size
    :return: The Blurred image
    """
    border = cv2.BORDER_REPLICATE
    kernel = cv2.getGaussianKernel(kernel_size, get_sigma(kernel_size=kernel_size))
    in_image = cv2.filter2D(in_image, -1, kernel, borderType=border)
    in_image = cv2.filter2D(in_image, -1, np.transpose(kernel), borderType=border)
    return in_image


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    g_ker = get_gaussian_kernel(5) * 4
    result_pyr = []
    gaussian_pyr = gaussianPyr(img, levels)
    for i in range(levels - 1):
        lap_level = gaussian_pyr[i] - gaussExpand(gaussian_pyr[i + 1], g_ker)
        result_pyr.append(lap_level.copy())
    result_pyr.append(gaussian_pyr[-1])
    return result_pyr


def get_gaussian_kernel(kernel_size: int):
    sigma = get_sigma(kernel_size=kernel_size, is_round=True)
    g_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    g_kernel = g_kernel * g_kernel.transpose()
    return g_kernel


def gaussExpand(img: np.ndarray, gs_k: np.ndarray) -> np.ndarray:
    """
    Expands a Gaussian pyramid level one step up
    :param img: Pyramid image at a certain level
    :param gs_k: The kernel to use in expanding
    :return: The expanded level
    """
    img_x_shape = img.shape[0]
    img_y_shape = img.shape[1]
    gauss_expand = np.zeros((2 * img_x_shape, 2 * img_y_shape))
    gauss_expand[::2, ::2] = img
    border = cv2.BORDER_REPLICATE
    gauss_expand = cv2.filter2D(gauss_expand, -1, gs_k, borderType=border)
    return gauss_expand


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    pyr = lap_pyr.copy()
    guss = get_gaussian_kernel(5) * 4
    current_layer = lap_pyr[-1]
    for i in range(len(pyr) - 2, -1, -1):
        current_layer = pyr[i] + gaussExpand(current_layer, guss)
    return current_layer


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    img_1_x_shape = img_1.shape[0]
    img_1_y_shape = img_1.shape[1]
    img_2_x_shape = img_2.shape[0]
    img_2_y_shape = img_2.shape[1]
    mask_x_shape = mask.shape[0]
    mask_y_shape = mask.shape[1]
    img_1 = img_1[0: 2 ** levels * int(img_1_x_shape / 2 ** levels), 0: 2 ** levels * int(img_1_y_shape / 2 ** levels)]
    img_2 = img_2[0: 2 ** levels * int(img_2_x_shape / 2 ** levels), 0: 2 ** levels * int(img_2_y_shape / 2 ** levels)]
    mask = mask[0: 2 ** levels * int(mask_x_shape / 2 ** levels), 0: 2 ** levels * int(mask_y_shape / 2 ** levels)]
    im_blend = np.zeros(img_1.shape)

    # rgb
    if (len(img_1.shape) == 3) or (len(img_2.shape) == 3):
        for color in range(3):
            part_im1, part_im2 = img_1[:, :, color], img_2[:, :, color]
            im_blend[:, :, color] = pyrBlend_helper(part_im1, part_im2, mask[:, :, color], levels)

    # grayscale
    else:
        im_blend = pyrBlend_helper(img_1, img_2, mask, levels)

    naive_blend = (img_2 * (1 - mask)) + (img_1 * mask)
    return naive_blend, im_blend


def pyrBlend_helper(img_1: np.ndarray, img_2: np.ndarray, mask: np.ndarray, levels: int) -> np.ndarray:
    """
        Blends two images using PyramidBlend method
        :param img_1: Image 1
        :param img_2: Image 2
        :param mask: Blend mask
        :param levels: Pyramid depth
        :return:  Blended Image
        """
    l1 = laplaceianReduce(img_1, levels)
    l2 = laplaceianReduce(img_2, levels)
    g = gaussianPyr(mask, levels)
    Lout = []
    for i in range(levels):
        Lout.append((l1[i] * g[i]) + (l2[i] * (1 - g[i])))
    im_blend = np.clip(laplaceianExpand(Lout), 0, 1)
    return im_blend
