import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from time import sleep


def flick(x):
    pass


def selectframes(videofile):
    cv.namedWindow('image')
    cv.moveWindow('image', 0, 0)

    cap = cv.VideoCapture(videofile)
    fbufferRGB = []

    nframes = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv.CAP_PROP_FPS)
    fwidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    fheight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    i = 0

    cv.createTrackbar('Frame', 'image', 0, nframes - 1, flick)
    cv.setTrackbarPos('Frame', 'image', 0)

    status = 'stay'
    while True:
        if i == nframes - 1:
            i = 0
        cap.set(cv.CAP_PROP_POS_FRAMES, i)
        ret, im = cap.read()
        res = cv.resize(im, (fwidth // 4, fheight // 4), interpolation=cv.INTER_AREA)
        cv.imshow('image', res)
        status = {ord('s'): 'stay', ord('S'): 'stay',
                  ord('w'): 'play', ord('W'): 'play',
                  ord('a'): 'prev_frame', ord('A'): 'prev_frame',
                  ord('d'): 'next_frame', ord('D'): 'next_frame',
                  ord('m'): 'snap', ord('M'): 'snap',
                  -1: status,
                  27: 'exit'}[cv.waitKey(10)]

        if status == 'play':
            sleep((0.1 - fps / 1000.0) ** 21021)
            i += 1
            cv.setTrackbarPos('Frame', 'image', i)
            continue
        if status == 'stay':
            i = cv.getTrackbarPos('Frame', 'image')
        if status == 'exit':
            break
        if status == 'prev_frame':
            i -= 1
            cv.setTrackbarPos('Frame', 'image', i)
            status = 'stay'
        if status == 'next_frame':
            i += 1
            cv.setTrackbarPos('Frame', 'image', i)
            status = 'stay'
        if status == 'snap':
            fbufferRGB.append(im)
            i += 1
            cv.setTrackbarPos('Frame', 'image', i)
            status = 'stay'

    cap.release()
    cv.destroyAllWindows()
    return fwidth, fheight, fbufferRGB


def fit_model(data_sample):
    """
    Ajusta el modelo recta-parábola mediante least squares.
    data_sample es el conjunto de datos escogidos para el ajuste inicial (3 o cuatro puntos).
    data_sample[0,:] => X, data_sample[1,:] => Y, data_sample[2,:] => FN
    ar, br => parámetros de la recta, ap, bp, cp => parámetros de la parábola
    """
    rec = np.polyfit(data_sample[2, :], data_sample[0, :], 1)
    par = np.polyfit(data_sample[2, :], data_sample[1, :], 2)
    return rec[0], rec[1], par[0], par[1], par[2]


def evaluate_model(data_rest, pars, inlier_threshold):
    """
    Evalua que puntos (datos) restantes son inliers.
    pars[0] = ar, pars[1] = br, pars[2] = ap, pars[3] = bp, pars[4] = cp
    """
    dist_recta = np.abs(data_rest[0, :] - pars[0] * data_rest[2, :] - pars[1])
    dist_parab = np.abs(
            data_rest[1, :] - pars[2] * (data_rest[2, :] ** 2) - pars[3] * data_rest[2, :] - pars[4])
    inliers = data_rest[:, dist_recta + dist_parab <= inlier_threshold]
    return inliers


def ransac(data, fit_fn, evaluate_fn, max_iters=1000, samples_to_fit=3, inlier_threshold=0.01, min_inliers=10):
    best_model = None
    best_pars = None
    best_model_performance = 0

    num_samples = data.shape[1]

    for ci in range(max_iters):
        sample = np.random.choice(num_samples, size=samples_to_fit, replace=False)
        data_sample = data[:, sample]
        data_rest = np.delete(data, sample, 1)
        model_params = fit_fn(data_sample)
        inliers = evaluate_fn(data_rest, model_params, inlier_threshold)
        num_inliers = inliers.shape[1]
        # print(num_inliers)

        if num_inliers < min_inliers:
            continue

        if num_inliers > best_model_performance:
            best_model = inliers
            best_pars = model_params
            best_model_performance = num_inliers

    return best_model, best_pars


if __name__ == "__main__":
    frameWidth, frameHeight, fblist = selectframes('video-02.mp4')
    frameCount = len(fblist)
    frameBufferRGB = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))
    frameBufferGray = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint8'))

    for i in range(frameCount):
        frameBufferRGB[i] = fblist[i]
        frameBufferGray[i] = cv.cvtColor(fblist[i], cv.COLOR_BGR2GRAY)

    kpts_list = []
    step = 2
    for i in range(step, frameCount - step):
        img1 = cv.absdiff(frameBufferGray[i], frameBufferGray[i - step])
        ret, img1 = cv.threshold(img1, 36, 255, cv.THRESH_BINARY)
        img2 = cv.absdiff(frameBufferGray[i + step], frameBufferGray[i - step])
        ret, img2 = cv.threshold(img2, 36, 255, cv.THRESH_BINARY_INV)
        img3 = cv.absdiff(frameBufferGray[i], frameBufferGray[i + step])
        ret, img3 = cv.threshold(img3, 36, 255, cv.THRESH_BINARY)
        img4 = cv.bitwise_and(img1, img2)
        img4 = cv.bitwise_and(img3, img4)

        imblobs = cv.bitwise_not(img4)
        params = cv.SimpleBlobDetector_Params()
        # Filter by Area
        params.filterByArea = True
        params.minArea = 150
        # params.maxArea = 10000

        # Filter by Circularity
        params.filterByCircularity = False
        # params.minCircularity = 0.1

        # Filter by Convexity
        params.filterByConvexity = False
        # params.minConvexity = 0.5

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0.3

        # Detect blobs.
        detector = cv.SimpleBlobDetector_create(params)
        kpts = detector.detect(imblobs)
        kpts_list.append(kpts)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    #     img4wkp = cv.drawKeypoints(frameBufferRGB[i], kpts, np.array([]), (0, 0, 255),
    #                                cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #
    #     cv.rectangle(img4wkp, (10, 2), (100, 20), (255, 255, 255), -1)
    #     cv.putText(img4wkp, str(i), (15, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
    #     res = cv.resize(img4wkp, (frameWidth // 2, frameHeight // 2), interpolation=cv.INTER_AREA)
    #     cv.imshow('frame delta', res)
    #     keyboard = cv.waitKey(250)
    #     if keyboard == 27:
    #         break
    # cv.destroyAllWindows()

    fn = []
    kp = []
    xp = []
    yp = []
    for i in range(len(kpts_list)):
        pts = [p.pt for p in kpts_list[i]]
        nkp = 0
        for xypt in pts:
            fn.append(i)
            kp.append(nkp)
            # xp.append(round(xypt[0]))
            # yp.append(round(xypt[1]))
            xp.append(xypt[0])
            yp.append(xypt[1])
            nkp += 1

    plt.subplot(2, 1, 1)
    plt.scatter(fn, xp, 20)
    plt.title('Frame contra coordenada X')

    plt.subplot(2, 1, 2)
    plt.scatter(fn, yp, 20)
    plt.title('Frame contra coordenada Y')

    plt.tight_layout()
    plt.show()

    xpn = np.array(xp, dtype=np.float64)
    xpn = xpn/xpn.max()
    print(xpn.shape)
    ypn = np.array(yp, dtype=np.float64)
    ypn = ypn/ypn.max()
    data = np.zeros((4, len(fn)))
    data[0, :] = xpn
    data[1, :] = ypn
    data[2, :] = fn
    data[3, :] = kp
    best_model, best_pars = ransac(data, fit_model, evaluate_model)
    print(best_model)
