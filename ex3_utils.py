import math
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt

MY_ID = 208980359
MIN_MSE = 1000
STEPS = 100
FULL_DEGREES = 360
WIN = 13
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
    k = 0
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

            A = np.array([[sum(NX[k] ** 2 for k in range(LEN_NX)), sum(NY[k] * NX[k] for k in range(LEN_NX))],
                          [sum(NX[k] * NY[k] for k in range(LEN_NX)), sum(NY[k] ** 2 for k in range(LEN_NY))]])

            B = np.array([[-1 * sum(NX[k] * NZ[k] for k in range(LEN_NX)),
                           -1 * sum(NY[k] * NZ[k] for k in range(LEN_NY))]]).reshape(2, 1)

            ev1, ev2 = np.linalg.eigvals(A)
            if ev2 < ev1:
                ev1, ev2 = ev2, ev1

            if ev2 >= ev1 > 1 and ev2 / ev1 < STEPS:
                velo = np.dot(np.linalg.pinv(A), B)
                v = velo[1][0]
                u = velo[0][0]
                UV.append(np.array([u, v]))
            else:
                k += 1
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
        dx_median, dy_median = np.ma.median(np.ma.masked_where(UVS == np.zeros((2)), UVS), axis=(0, 1)).filled(0)
        pyr_1 = cv2.warpPerspective(pyr_1, get_base_kernel(x=dx_median, y=dy_median), pyr_1.shape[::-1])
        PTs, uv = opticalFlow(pyr_1, pyr_2, max(int(stepSize * math.pow(2, -level)), 1), winSize)
        if PTs.size == 0:
            continue
        converted_points = np.power(2, level) * PTs
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
    old, new = opticalFlow(im1, im2, 10, 5)
    # look at all the u,v we found
    for x in range(len(new)):
        t1 = new[x][0]
        t2 = new[x][1]
        t = get_base_kernel(x=t1, y=t2)
        # create a new image a transformation using u,v
        newimg = cv2.warpPerspective(im1, t, (im1.shape[1], im1.shape[0]))
        # find difference in image and keep track of the x,y that gives the smallest diff
        d = ((im2 - newimg) ** 2).sum()

        if d < diff:
            diff = d
            spot = x
            if diff == 0:
                print("break")
                break
    t1 = new[spot][0]
    t2 = new[spot][1]

    t = get_base_kernel(x=t1, y=t2)

    return t


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
    t = 0
    min_mse = MIN_MSE
    for t in range(FULL_DEGREES):
        tmp_t = np.array([[math.cos(t), -math.sin(t), 0], [math.sin(t), math.cos(t), 0], [0, 0, 1]], dtype=np.float64)
        img_by_t = cv2.warpPerspective(im1, tmp_t, im1.shape[::-1])
        mse = np.square(np.subtract(im2, img_by_t)).mean()
        if mse < min_mse:
            min_mse = mse
            tran_mat = tmp_t
            t = t
    rigid_mat = np.array([[math.cos(t), math.sin(t), 0], [-math.sin(t), math.cos(t), 0], [0, 0, 1]], dtype=np.float64)
    revers_img = cv2.warpPerspective(im2, rigid_mat, im2.shape[::-1])
    tran_mat = findTranslationLK(im1, revers_img)
    ty = tran_mat[1, 2]
    tx = tran_mat[0, 2]
    ans = np.array([[math.cos(t), -math.sin(t), tx], [math.sin(t), math.cos(t), ty], [0, 0, 1]], dtype=np.float64)

    return ans


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    win = WIN
    pad = win // 2
    im2pad = cv2.copyMakeBorder(im2, pad, pad, pad, pad, cv2.BORDER_REPLICATE, None, value=0)

    # getting 4 x and y points to be the middle of the window
    # the points are 1/5, 2/5 ... of the length and height
    I = []
    J = []
    for x in range(1, 5):
        I.append((im1.shape[0] // 5) * x)
        J.append((im1.shape[1] // 5) * x)

    corr_listt = [(np.array([0]), 0, 0)]
    for x in range(len(I)):
        for y in range(len(J)):
            # getting a template to match
            windowa = im1[I[x] - pad:I[x] + pad + 1, J[y] - pad:J[y] + pad + 1]
            a = windowa.reshape(1, win * win)
            aT = a.T
            big = [(np.array([0]), 0, 0)]
            # going through the other pic to match the template
            for i in range(0, im2.shape[0]):
                for j in range(0, im2.shape[1]):
                    if (i + pad + win) < im2pad.shape[0] and (j + pad + win) < im2pad.shape[1]:
                        windowb = im2pad[i + pad:i + pad + win, j + pad:j + pad + win]
                        b = windowb.reshape(1, win * win)
                        bT = b.T
                        top = np.dot(a, bT)
                        bottom = np.dot(a, aT) + np.dot(b, bT)
                        # finding the correlation between the template and this window
                        # if it is bigger than the first value in list big clear big and put it in with the x y values of im2
                        # if it is equal to the first value add it to the list and put it in with the x y values of im2
                        if bottom != 0:
                            corr = top / bottom
                            if corr > big[0][0]:
                                big.clear()
                                big.insert(0, (corr, i, j))
                            elif corr == big[0][0]:
                                big.insert(0, (corr, i, j))
            # after checking this template check if the first value in big is bigger than the first value in corr_lisst
            # if so clear corr_listt and copy the values from big to corr_listt and add the x y vaues of the original image
            # if it equals copy the values from big to corr_listt and add the x y vaues of the original image
            if big[0][0][0] > corr_listt[0][0][0]:
                corr_listt.clear()
                for m in range(len(big)):
                    corr_listt.append((big[m], (I[x], J[y])))
            if big[0][0][0] == corr_listt[0][0][0]:
                for m in range(len(big)):
                    corr_listt.append((big[m], (I[x], J[y])))

    dif = float("inf")
    spot = -1
    # go through all values in the cor list and find the u v by finding the difference between im1 xy and im2 xy
    for x in range(len(corr_listt)):

        t1 = corr_listt[x][1][0] - corr_listt[x][0][1]  # u
        t2 = corr_listt[x][1][1] - corr_listt[x][0][2]  # v
        # create a new img with the found transformation
        t = get_base_kernel(x=t1, y=t2)
        new = cv2.warpPerspective(im1, t, (im1.shape[1], im1.shape[0]))
        # find the difference between new and im2 if smaller than diff update diff and spot
        d = ((im2 - new) ** 2).sum()
        if d < dif:
            dif = d
            spot = x
            if dif == 0:
                break
    # take the values from corrlist that has the smallest diff and return the transformation
    t1 = corr_listt[spot][1][0] - corr_listt[spot][0][1]  # u
    t2 = corr_listt[spot][1][1] - corr_listt[spot][0][2]  # v
    t = get_base_kernel(x=t1, y=t2)
    return t


def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.
    """
    win = 5
    pad = win // 2

    im2pad = cv2.copyMakeBorder(im2, pad, pad, pad, pad, cv2.BORDER_REPLICATE, None, value=0)

    # getting 4 x and y points to be the middle of the window
    # the points are 1/5, 2/5 ... of the length and height
    I = []
    J = []
    for x in range(1, 5):
        I.append((im1.shape[0] // 5) * x)
        J.append((im1.shape[1] // 5) * x)

    corr_listt = [(np.array([0]), 0, 0)]
    for x in range(len(I)):
        for y in range(len(J)):
            # getting a template to match
            windowa = im1[I[x] - pad:I[x] + pad + 1, J[y] - pad:J[y] + pad + 1]
            a = windowa.reshape(1, win * win)
            aT = a.T
            big = [(np.array([0]), 0, 0)]
            for i in range(0, im2.shape[0]):
                for j in range(0, im2.shape[1]):
                    if (i + pad + win) < im2pad.shape[0] and (j + pad + win) < im2pad.shape[1]:
                        windowb = im2pad[i + pad:i + pad + win, j + pad:j + pad + win]
                        b = windowb.reshape(1, win * win)
                        bT = b.T
                        top = np.dot(a, bT)
                        bottom = np.dot(a, aT) + np.dot(b, bT)
                        # finding the correlation between the template and this window
                        # if it is bigger than the first value in list big clear big and put it in with the x y values of im2
                        # if it is equal to the first value add it to the list and put it in with the x y values of im2
                        if bottom != 0:
                            corr = top / bottom
                            if corr > big[0][0]:
                                big.clear()
                                big.insert(0, (corr, i, j))
                            elif corr == big[0][0]:
                                big.insert(0, (corr, i, j))
            # after checking this template check if the first value in big is bigger than the first value in corr_lisst
            # if so clear corr_listt and copy the values from big to corr_listt and add the x y vaues of the original image
            # if it equals copy the values from big to corr_listt and add the x y vaues of the original image
            if big[0][0][0] > corr_listt[0][0][0]:
                corr_listt.clear()
                for m in range(len(big)):
                    corr_listt.append((big[m], (I[x], J[y])))
            if big[0][0][0] == corr_listt[0][0][0]:
                for m in range(len(big)):
                    corr_listt.append((big[m], (I[x], J[y])))

    spot = -1
    diff = float("inf")
    # go through all values in the cor_list and find the u v and theta
    # by finding the difference between im1 xy and im2 xy
    for n in range(len(corr_listt)):
        x = corr_listt[n][1][0] - corr_listt[n][0][1]
        y = corr_listt[n][1][1] - corr_listt[n][0][2]

        if y != 0:
            theta = np.arctan(x / y)
        else:
            theta = 0
        # create a new img with the found transformation
        t = np.array([[np.cos(theta), -np.sin(theta), x], [np.sin(theta), np.cos(theta), y], [0, 0, 1]], dtype=np.float)
        new = cv2.warpPerspective(im1, t, im1.shape[::-1])
        # find the difference between new and im2 if smaller than diff update diff and spot
        d = ((im2 - new) ** 2).sum()
        if d < diff:
            diff = d
            spot = n
        if diff == 0:
            break
    # take the values from corrlist that has the smallest diff and return the transformation
    x = corr_listt[spot][1][0] - corr_listt[spot][0][1]
    y = corr_listt[spot][1][1] - corr_listt[spot][0][2]

    theta = 0
    if y != 0:
        theta += np.arctan(x / y)


    return np.array([[np.cos(theta), -np.sin(theta), x], [np.sin(theta), np.cos(theta), y], [0, 0, 1]], dtype=np.float)


def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    new = np.zeros((im1.shape[0], im1.shape[1]))
    Tinv = np.linalg.inv(T)
    print(T, "\n", Tinv)
    for i in range(im2.shape[0]):
        for j in range(im2.shape[1]):
            arr = np.array([i, j, 1])
            newarr = Tinv @ arr
            x1 = np.floor(newarr[0]).astype(int)
            x2 = np.ceil(newarr[0]).astype(int)
            x3 = round(newarr[0] % 1, 3)
            y1 = np.floor(newarr[1]).astype(int)
            y2 = np.ceil(newarr[1]).astype(int)
            y3 = round(newarr[1] % 1, 3)

            if x1 >= 0 and y1 >= 0 and x1 < im1.shape[0] and y1 < im1.shape[1]:
                new[i][j] += (1 - x3) * (1 - y3) * im1[x1][y1]

            if x2 >= 0 and y1 >= 0 and x2 < im1.shape[0] and y1 < im1.shape[1]:
                new[i][j] += x3 * (1 - y3) * im1[x2][y1]

            if x1 >= 0 and y2 >= 0 and x1 < im1.shape[0] and y2 < im1.shape[1]:
                new[i][j] += (1 - x3) * y3 * im1[x1][y2]

            if x2 >= 0 and y2 >= 0 and x2 < im1.shape[0] and y2 < im1.shape[1]:
                new[i][j] += x3 * y3 * im1[x2][y2]

    plt.imshow(new)
    plt.show()
    return new


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
    img = img[0: np.power(2, levels) * int(img.shape[0] / np.power(2, levels)),
          0: np.power(2, levels) * int(img.shape[1] / np.power(2, levels))]

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
    sigma = get_sigma(kernel_size=kernel_size)
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    in_image = cv2.filter2D(in_image, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    in_image = cv2.filter2D(in_image, -1, np.transpose(kernel), borderType=cv2.BORDER_REPLICATE)
    return in_image


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    pyr = []
    g_ker = gaussian_Kernel(5)
    g_ker *= 4
    gaussian_pyr = gaussianPyr(img, levels)
    for i in range(levels - 1):
        extend_level = gaussExpand(gaussian_pyr[i + 1], g_ker)
        lap_level = gaussian_pyr[i] - extend_level
        pyr.append(lap_level.copy())
    pyr.append(gaussian_pyr[-1])
    return pyr


def gaussian_Kernel(kernel_size: int):
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
    expand = np.zeros((img.shape[0] * 2, img.shape[1] * 2))
    expand[::2, ::2] = img
    expand = cv2.filter2D(expand, -1, gs_k, borderType=cv2.BORDER_REPLICATE)
    return expand


def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    pyr_updated = lap_pyr.copy()
    guss_k = gaussian_Kernel(5) * 4
    cur_layer = lap_pyr[-1]
    for i in range(len(pyr_updated) - 2, -1, -1):
        cur_layer = gaussExpand(cur_layer, guss_k) + pyr_updated[i]
    return cur_layer


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> (np.ndarray, np.ndarray):
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    assert (img_1.shape == img_2.shape)

    img_1 = img_1[0: np.power(2, levels) * int(img_1.shape[0] / np.power(2, levels)),
            0: np.power(2, levels) * int(img_1.shape[1] / np.power(2, levels))]
    img_2 = img_2[0: np.power(2, levels) * int(img_2.shape[0] / np.power(2, levels)),
            0: np.power(2, levels) * int(img_2.shape[1] / np.power(2, levels))]
    mask = mask[0: np.power(2, levels) * int(mask.shape[0] / np.power(2, levels)),
           0: np.power(2, levels) * int(mask.shape[1] / np.power(2, levels))]

    im_blend = np.zeros(img_1.shape)
    if len(img_1.shape) == 3 or len(img_2.shape) == 3:  # the image is RGB
        for color in range(3):
            part_im1 = img_1[:, :, color]
            part_im2 = img_2[:, :, color]
            part_mask = mask[:, :, color]
            im_blend[:, :, color] = pyrBlend_helper(part_im1, part_im2, part_mask, levels)

    else:  # the image is grayscale
        im_blend = pyrBlend_helper(img_1, img_2, mask, levels)

    # Naive blend
    naive_blend = mask * img_1 + (1 - mask) * img_2

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
    L1 = laplaceianReduce(img_1, levels)
    L2 = laplaceianReduce(img_2, levels)
    Gm = gaussianPyr(mask, levels)
    Lout = []
    for k in range(levels):
        curr_lup = Gm[k] * L1[k] + (1 - Gm[k]) * L2[k]
        Lout.append(curr_lup)
    im_blend = laplaceianExpand(Lout)
    im_blend = np.clip(im_blend, 0, 1)

    return im_blend
