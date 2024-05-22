import os
import cv2
import numpy as np
from tqdm import tqdm
from metrics import PSNR


class Binarization:
    def __init__(self, window_size, alpha, sigma0):
        self.window_size = window_size
        self.alpha = alpha
        self.sigma0 = sigma0

    @staticmethod
    def prepare_image(image):
        if len(image.shape) == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        elif len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    @staticmethod
    def otsu(image):
        hist, bins = np.histogram(image, bins=256, range=(0, 255))
        norm_hist = hist / np.sum(hist)

        max_criterion = 0  # max between-class variance
        threshold = 0
        for k in range(256):
            w0 = np.sum(norm_hist[:k])
            w1 = np.sum(norm_hist[k:])

            mu0 = np.sum(np.arange(k) * norm_hist[:k]) / w0 if w0 > 0 else 0
            mu1 = np.sum(np.arange(k, 256) * norm_hist[k:]) / w1 if w1 > 0 else 0

            criterion = w0 * w1 * (mu1 - mu0) ** 2

            if criterion > max_criterion:
                max_criterion = criterion
                threshold = k

        result = 255 * (image > threshold).astype(np.uint8)

        return result

    @staticmethod
    def unbalanced_otsu(image):
        hist, bins = np.histogram(image, bins=256, range=(0, 255))
        norm_hist = hist / np.sum(hist)

        max_criterion = -np.inf
        threshold = 0
        for k in range(256):
            w0 = np.sum(norm_hist[:k])
            w1 = np.sum(norm_hist[k:])

            mu0 = np.sum(np.arange(k) * norm_hist[:k]) / w0 if w0 > 0 else 0
            mu1 = np.sum(np.arange(k, 256) * norm_hist[k:]) / w1 if w1 > 0 else 0

            var0 = np.sum((np.arange(k) - mu0) ** 2 * norm_hist[:k]) / w0 if w0 > 0 else 0  # squared
            var1 = np.sum((np.arange(k, 256) - mu1) ** 2 * norm_hist[k:]) / w1 if w1 > 0 else 0  # squared

            var_w = w0 * var0 + w1 * var1  # within-class variance (squared)
            criterion = 0
            if w0 != 0:
                criterion += w0 * np.log(w0)
            if w1 != 0:
                criterion += w1 * np.log(w1)
            criterion -= np.log(var_w) / 2
            if criterion > max_criterion:
                max_criterion = criterion
                threshold = k

        result = 255 * (image > threshold).astype(np.uint8)

        return result

    def niblack(self, image):
        padded_image = image.copy()
        padded_image = np.hstack([np.zeros((padded_image.shape[0], 1)), padded_image])
        padded_image = np.vstack([np.zeros((1, padded_image.shape[1])), padded_image])
        height, width = padded_image.shape

        top_bound = np.maximum(np.arange(-self.window_size // 2 + 1, height - 1 - self.window_size // 2), 0)
        bottom_bound = np.minimum(np.arange(self.window_size // 2 + 1, height + self.window_size // 2), height - 1)
        left_bound = np.maximum(np.arange(-self.window_size // 2 + 1, width - 1 - self.window_size // 2), 0)
        right_bound = np.minimum(np.arange(self.window_size // 2 + 1, width + self.window_size // 2), width - 1)

        cum_image = np.cumsum(np.cumsum(padded_image.astype(np.int64), axis=1), axis=0)
        cum_image_2 = np.cumsum(np.cumsum(np.power(padded_image.astype(np.int64), 2), axis=1), axis=0)

        sum_1 = (cum_image[bottom_bound[:, None], right_bound[None, :]]
                 - cum_image[bottom_bound[:, None], left_bound[None, :]]
                 - cum_image[top_bound[:, None], right_bound[None, :]]
                 + cum_image[top_bound[:, None], left_bound[None, :]])

        sum_2 = (cum_image_2[bottom_bound[:, None], right_bound[None, :]]
                 - cum_image_2[bottom_bound[:, None], left_bound[None, :]]
                 - cum_image_2[top_bound[:, None], right_bound[None, :]]
                 + cum_image_2[top_bound[:, None], left_bound[None, :]])

        areas = (bottom_bound[:, None] - top_bound[:, None]) * (right_bound[None, :] - left_bound[None, :])

        means = sum_1 / areas
        std = np.sqrt(sum_2 / areas - means ** 2)
        threshold = np.clip(means + self.alpha * std, 0, 255)

        result = 255 * (image > threshold).astype(np.uint8)

        return result

    def adaptive_niblack(self, image):
        padded_image = image.copy()
        padded_image = np.hstack([np.zeros((padded_image.shape[0], 1)), padded_image])
        padded_image = np.vstack([np.zeros((1, padded_image.shape[1])), padded_image])
        height, width = padded_image.shape

        cum_image = np.cumsum(np.cumsum(padded_image.astype(np.int64), axis=1), axis=0)
        cum_image_2 = np.cumsum(np.cumsum(np.power(padded_image.astype(np.int64), 2), axis=1), axis=0)

        threshold = np.empty(image.shape).astype(np.int16)
        threshold.fill(-1)

        completed = False
        window_size = self.window_size
        while window_size <= np.min(image.shape):
            top_bound = np.maximum(np.arange(-window_size // 2 + 1, height - 1 - window_size // 2), 0)
            bottom_bound = np.minimum(np.arange(window_size // 2 + 1, height + window_size // 2), height - 1)
            left_bound = np.maximum(np.arange(-window_size // 2 + 1, width - 1 - window_size // 2), 0)
            right_bound = np.minimum(np.arange(window_size // 2 + 1, width + window_size // 2), width - 1)

            sum_1 = (cum_image[bottom_bound[:, None], right_bound[None, :]]
                     - cum_image[bottom_bound[:, None], left_bound[None, :]]
                     - cum_image[top_bound[:, None], right_bound[None, :]]
                     + cum_image[top_bound[:, None], left_bound[None, :]])

            sum_2 = (cum_image_2[bottom_bound[:, None], right_bound[None, :]]
                     - cum_image_2[bottom_bound[:, None], left_bound[None, :]]
                     - cum_image_2[top_bound[:, None], right_bound[None, :]]
                     + cum_image_2[top_bound[:, None], left_bound[None, :]])

            areas = (bottom_bound[:, None] - top_bound[:, None]) * (right_bound[None, :] - left_bound[None, :])

            means = sum_1 / areas
            std = np.sqrt(sum_2 / areas - means ** 2)
            fill_condition = (std > self.sigma0) & (threshold == -1)
            threshold[fill_condition] = np.clip(means[fill_condition] + self.alpha * std[fill_condition], 0, 255)

            if np.count_nonzero(threshold == -1) > 0:
                window_size = window_size * 2 - 1
            else:
                completed = True
                break

        if not completed:
            print('The image doesn\'t have any remarkable structures')
            return np.zeros_like(image)

        result = 255 * (image > threshold).astype(np.uint8)

        return result

    def test(self, images_path, labels_path):
        print('\nTesting...')

        filenames = os.listdir(images_path)
        assert len(set(filenames).difference(set(os.listdir(labels_path)))) == 0, 'Set of images != set of labels'

        for method in [Binarization.otsu, Binarization.unbalanced_otsu, self.niblack, self.adaptive_niblack]:
            metrics = list()

            print(method.__name__)
            print(f'filename \t\t PSNR')
            for filename in filenames:
                image_path = os.path.join(images_path, filename)
                label_path = os.path.join(labels_path, filename)

                image = Binarization.prepare_image(cv2.imread(image_path))
                label = Binarization.prepare_image(cv2.imread(label_path))

                prediction = method(image)
                psnr = PSNR(prediction, label)
                metrics.append(psnr)

                print(f'{filename} \t\t {psnr}')

            print(f'Mean PSNR: {sum(metrics) / len(metrics)}\n')

