from ex3_utils import *
import time


# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------


def lkDemo(img_path):
    print("LK Demo")

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, -.2],
                  [0, 1, -.1],
                  [0, 0, 1]], dtype=float)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])
    st = time.time()
    pts, uv = opticalFlow(img_1.astype(float), img_2.astype(float), step_size=20, win_size=5)
    et = time.time()

    print("Time: {:.4f}".format(et - st))
    print(np.median(uv,0))
    print(np.mean(uv,0))

    displayOpticalFlow(img_2, pts, uv)


def hierarchicalkDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Hierarchical LK Demo")
    img_1 = cv2.imread(img_path)
    img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, 3],
                  [0, 1, -3],
                  [0, 0, 1]], dtype=np.float32)
    img_2 = cv2.warpPerspective(img_1, t, img_1.shape[::-1])

    start = time.time()
    uvs = opticalFlowPyrLK(img_1.astype(np.float32), img_2.astype(
        np.float32), 10, stepSize=20, winSize=5)
    end = time.time()

    pts = np.where(np.not_equal(uvs[:, :], np.zeros((2))))
    uvs = uvs[pts[0], pts[1]]
    print("Time: {:.4f}".format(end - start))
    print(np.median(uvs, 0))
    print(np.mean(uvs, 0))
    plt.imshow(img_2, cmap='gray')
    plt.quiver(pts[1], pts[0], uvs[:, 0], uvs[:, 1], color='r')

    plt.show()


def compareLK(img_path):
    """
    ADD TEST
    Compare the two results from both functions.
    :param img_path: Image input
    :return:
    """
    print("Compare LK & Hierarchical LK")

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.5, fy=0.5)
    t = np.array([[1, 0, 3],
                  [0, 1, -3],
                  [0, 0, 1]], dtype=np.float32)
    img_2 = cv2.warpPerspective(
        img_1, t, img_1.shape[::-1], flags=cv2.INTER_LINEAR)
    st = time.time()
    _, uv_naive = opticalFlow(img_1.astype(np.float32), img_2.astype(
        np.float32), step_size=20, win_size=5)
    et = time.time()

    print("Compare LK & Hierarchical LK")
    print("Time of naive method: {:.4f}".format(et - st))
    print('median of naive method:', np.median(uv_naive, 0))
    print('mean of naive method:', np.mean(uv_naive, 0))

    st = time.time()
    uv_pyr = opticalFlowPyrLK(img_1.astype(np.float32), img_2.astype(
        np.float32), 7, stepSize=20, winSize=5)
    et = time.time()
    median_pyr = np.ma.median(np.ma.masked_where(
        uv_pyr == np.zeros((2)), uv_pyr), axis=(0, 1)).filled(0)
    mean_pyr = np.ma.mean(np.ma.masked_where(
        uv_pyr == np.zeros((2)), uv_pyr), axis=(0, 1)).filled(0)
    print("Time of hierarchical method: {:.4f}".format(et - st))
    print('median of hierarchical method:', median_pyr)
    print('mean of hierarchical method:', mean_pyr)
    ground_truth = np.array([3, -3])
    diff_naive = np.power(np.median(uv_naive, 0) - ground_truth, 2).sum() / 2
    diff_pyr = np.power(median_pyr - ground_truth, 2).sum() / 2
    print('accuracy improved by:', diff_naive - diff_pyr)


def displayOpticalFlow(img: np.ndarray, pts: np.ndarray, uvs: np.ndarray):
    plt.imshow(img, cmap='gray')
    plt.quiver(pts[:, 0], pts[:, 1], uvs[:, 0], uvs[:, 1], color='r')

    plt.show()


# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------


def imageWarpingDemo(img_path):
    """
    ADD TEST
    :param img_path: Image input
    :return:
    """
    print("Image Warping Demo")

    def demo(im1, im2, t, finding_func):
        st = time.time()
        res = finding_func(im1.astype(np.float32), im2.astype(
            np.float32))
        et = time.time()

        print("Time: {:.4f}".format(et - st))
        print('Translation found:')
        print(res)
        print('SE:')
        # print(np.power(t-res, 2).mean())

    img_1 = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
    img_1 = cv2.resize(img_1, (0, 0), fx=.1, fy=0.1)

    t = np.array([[1, 0, -5],
                  [0, 1, 8],
                  [0, 0, 1]],
                 dtype=np.float32)

    img_2 = cv2.warpPerspective(
        img_1, t, img_1.shape[::-1])
    theta = np.deg2rad(-20)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    t2 = np.array([[cos_t, sin_t, 12],
                  [-sin_t, cos_t, -12.5]],
                  dtype=np.float32)

    img_3 = cv2.warpAffine(
        img_1, t2, img_1.shape[::-1])

    print("Compare LK & Hierarchical LK (Movement)")
    print("Original transformation:")
    print(t)

    print("-LK Results:")
    demo(img_1, img_2, t, findTranslationLK)

    print("-Correlation Results:")
    demo(img_1, img_2, t, findTranslationCorr)

    print("Compare LK & Hierarchical LK (Rigid)")
    print("Original transformation:")
    print(t2)

    print("-LK Results:")
    demo(img_1, img_3, t2, findRigidLK)

    print("-Correlation Results:")
    demo(img_1, img_3, t2, findRigidCorr)


# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def pyrGaussianDemo(img_path):
    print("Gaussian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 4
    gau_pyr = gaussianPyr(img, lvls)

    h, w = gau_pyr[0].shape[:2]
    canv_h = h
    widths = np.cumsum([w // (2 ** i) for i in range(lvls)])
    widths = np.hstack([0, widths])
    canv_w = widths[-1]
    canvas = np.zeros((canv_h, canv_w, 3))

    for lv_idx in range(lvls):
        h = gau_pyr[lv_idx].shape[0]
        canvas[:h, widths[lv_idx]:widths[lv_idx + 1], :] = gau_pyr[lv_idx]

    plt.imshow(canvas)
    plt.show()


def pyrLaplacianDemo(img_path):
    print("Laplacian Pyramid Demo")

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY) / 255
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    lvls = 7

    lap_pyr = laplaceianReduce(img, lvls)
    re_lap = laplaceianExpand(lap_pyr)

    f, ax = plt.subplots(2, lvls + 1)
    plt.gray()
    for i in range(lvls):
        ax[0, i].imshow(lap_pyr[i])
        ax[1, i].hist(lap_pyr[i].ravel(), 256, [lap_pyr[i].min(), lap_pyr[i].max()])

    ax[0, -1].set_title('Original Image')
    ax[0, -1].imshow(re_lap)
    ax[1, -1].hist(re_lap.ravel(), 256, [0, 1])
    plt.show()


def blendDemo():
    im1 = cv2.cvtColor(cv2.imread('input/sunset.jpg'), cv2.COLOR_BGR2RGB) / 255
    im2 = cv2.cvtColor(cv2.imread('input/cat.jpg'), cv2.COLOR_BGR2RGB) / 255
    mask = cv2.cvtColor(cv2.imread('input/mask_cat.jpg'), cv2.COLOR_BGR2RGB) / 255

    n_blend, im_blend = pyrBlend(im1, im2, mask, 4)

    f, ax = plt.subplots(2, 3)
    ax[0, 0].imshow(im1)
    ax[0, 1].imshow(im2)
    ax[0, 2].imshow(mask)
    ax[1, 0].imshow(n_blend)
    ax[1, 1].imshow(np.abs(n_blend - im_blend))
    ax[1, 2].imshow(im_blend)

    plt.show()

    cv2.imwrite('sunset_cat.png', cv2.cvtColor((im_blend * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


def main():
    print("ID:", myID())

    img_path = 'input/boxMan.jpg'
    lkDemo(img_path)
    hierarchicalkDemo(img_path)
    compareLK(img_path)

    imageWarpingDemo(img_path)

    pyrGaussianDemo('input/pyr_bit.jpg')
    pyrLaplacianDemo('input/pyr_bit.jpg')
    blendDemo()


if __name__ == '__main__':
    main()
