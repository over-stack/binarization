import os
import argparse
import cv2
from tqdm import tqdm
from methods import Binarization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, help='path to image directory')
    parser.add_argument('-o', type=str, help='output directory for bin image')
    parser.add_argument('-w', type=int, help='Niblack window size')
    parser.add_argument('-a', type=float, help='Niblack parameter')
    parser.add_argument('-s', type=float, help='modified Niblack parameter')
    parser.add_argument('--test', type=bool, default=False, help='test algorithms')
    parser.add_argument('--test_i', type=str, default='None', help='test images dir path')
    parser.add_argument('--test_l', type=str, default='None', help='test labels dir path')
    args = parser.parse_args()

    try:
        input_dir = args.i
        output_dir = args.o
        window_size = args.w
        alpha = args.a
        sigma0 = args.s
        test = args.test
        test_images_dir = args.test_i
        test_labels_dir = args.test_l

        print(args.__dict__)
    except Exception as e:
        print(e)
        return

    binarization = Binarization(window_size, alpha, sigma0)

    if test:
        binarization.test(test_images_dir, test_labels_dir)

    filenames = os.listdir(input_dir)
    for filename in tqdm(filenames):
        image_path = os.path.join(input_dir, filename)
        image = Binarization.prepare_image(cv2.imread(image_path))

        prediction = Binarization.otsu(image)
        prediction_path = os.path.join(output_dir, 'otsu', filename)
        cv2.imwrite(prediction_path, prediction)

        prediction = Binarization.unbalanced_otsu(image)
        prediction_path = os.path.join(output_dir, 'unbalanced_otsu', filename)
        cv2.imwrite(prediction_path, prediction)

        prediction = binarization.niblack(image)
        prediction_path = os.path.join(output_dir, 'niblack', filename)
        cv2.imwrite(prediction_path, prediction)

        prediction = binarization.adaptive_niblack(image)
        prediction_path = os.path.join(output_dir, 'adaptive_niblack', filename)
        cv2.imwrite(prediction_path, prediction)


if __name__ == '__main__':
    main()

