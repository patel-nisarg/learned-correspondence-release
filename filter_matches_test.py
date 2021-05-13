import pickle
import numpy as np
import cv2
from network import MyNetwork
from evaluate import test_pair
import matplotlib.pyplot as plt
import os


def match(im_1, im_2, n_kp=2000):
    sift = cv2.xfeatures2d.SIFT_create(
        nfeatures=n_kp, contrastThreshold=1e-5)

    cv_kp_1, cv_desc_1 = sift.detectAndCompute(im_1, None)
    cv_kp_2, cv_desc_2 = sift.detectAndCompute(im_2, None)

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches_all = matcher.match(cv_desc_1, cv_desc_2)

    scores = np.array([x.distance for x in matches_all])
    idx1 = np.array([x.queryIdx for x in matches_all])
    idx2 = np.array([x.trainIdx for x in matches_all])

    kp1_x = np.array([k.pt[0] for k in cv_kp_1])[idx1]
    kp1_y = np.array([k.pt[1] for k in cv_kp_1])[idx1]
    kp1 = np.stack((kp1_x, kp1_y), axis=1)

    kp2_x = np.array([k.pt[0] for k in cv_kp_2])[idx2]
    kp2_y = np.array([k.pt[1] for k in cv_kp_2])[idx2]
    kp2 = np.stack((kp2_x, kp2_y), axis=1)

    return kp1, kp2, scores


def run_model(mynet, kp1, kp2, shape_im_1, shape_im_2, focal_len_1=None, focal_len_2=None):
    # normalize coordinates: if focal lengths are not given, use simpler normalization
    norm_factor_1 = focal_len_1 if focal_len_1 is not None else np.max(shape_im_1[:2])
    norm_factor_2 = focal_len_2 if focal_len_2 is not None else np.max(shape_im_2[:2])
    kp1 = (kp1 - 0.5 * np.array(shape_im_1[:2])[::-1]) / norm_factor_1
    kp2 = (kp2 - 0.5 * np.array(shape_im_2[:2])[::-1]) / norm_factor_2
    # concatenate
    xs = np.expand_dims(np.concatenate((kp1, kp2), axis=1), 0)
    # run
    ys = test_pair(mynet, xs)
    return ys


with open('./demo/config.pickle', 'rb') as handle:
    config = pickle.load(handle)

mynet = MyNetwork(config)
mynet.load_model()

im_1 = cv2.imread('F:\Documents\SFM_Project\images\images\img0002.jpg')
im_2 = cv2.imread('F:\Documents\SFM_Project\images\images\img0008.jpg')

# get initial matches
kp1, kp2, _ = match(im_1, im_2)

# run model
ys = run_model(mynet, kp1, kp2, im_1.shape, im_2.shape)

# resort matches according to model output
sorted_idx = np.argsort(-ys).squeeze()
kp1 = kp1[sorted_idx, :]
kp2 = kp2[sorted_idx, :]
scores = ys.squeeze()[sorted_idx]

# filter out matches with negative scores
kp1 = kp1[scores > 0]
kp2 = kp2[scores > 0]
scores = scores[scores > 0]

top_N = 100

# pad image if required
if im_1.shape[1] > im_2.shape[1]:
    im_2 = np.concatenate((im_2, np.zeros((im_2.shape[0], im_1.shape[1] - im_2.shape[1], 3),
                                          dtype=np.uint8)), axis=1)
elif im_1.shape[1] < im_2.shape[1]:
    im_1 = np.concatenate((im_1, np.zeros((im_1.shape[0], im_2.shape[1] - im_1.shape[1], 3),
                                          dtype=np.uint8)), axis=1)

# plot
plt.imshow(np.concatenate((im_1[:, :, [2, 1, 0]], im_2[:, :, [2, 1, 0]]), axis=0))
plt.scatter(kp1[:top_N, 0], kp1[:top_N, 1], 0.5, c='limegreen', alpha=0.5)
plt.scatter(kp2[:top_N, 0], im_1.shape[0] + kp2[:top_N, 1], 0.5, c='limegreen', alpha=0.5)
plt.plot(np.stack((kp1[:top_N, 0], kp2[:top_N, 0])),
         np.stack((kp1[:top_N, 1], im_1.shape[0] + kp2[:top_N, 1])),
         c='limegreen',
         linewidth=0.1)
plt.axis('off')
plt.gcf().set_dpi(400)
plt.show()
