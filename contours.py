import argparse
import cv2
import numpy as np
import wolframalpha
from sklearn.externals import joblib
from skimage.feature import hog
from classifier import labels_to_image_type
from secrets import WOLFRAM_KEY


def get_contours(image):
    im = cv2.imread(image)

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    ret, im_th = cv2.threshold(im_gray, 110, 255, cv2.THRESH_BINARY_INV)

    _, contours, _ = cv2.findContours(im_th.copy(), cv2.RETR_TREE,
                                      cv2.CHAIN_APPROX_NONE)

    return im_th, contours


def predict(classifier, im_th, contours):
    sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    clf = joblib.load(classifier)

    rects = [cv2.boundingRect(contour) for contour in sorted_ctrs]

    predictions = []

    for rect in rects:
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        if not roi.size:
            continue
        roi = cv2.resize(roi, (45, 45), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        roi_hog_fd = hog(roi, orientations=6, pixels_per_cell=(15, 15),
                         cells_per_block=(1, 1), visualise=False)
        predictions.append(clf.predict(np.array([roi_hog_fd], 'float64')))

    return np.hstack(predictions).tolist()


def convert_to_equation(labels):
    client = wolframalpha.Client(WOLFRAM_KEY)
    equation = ''.join([labels_to_image_type[label] for label in labels])
    try:
        answer = next(client.query('Question: ' + equation).results).text
        print(equation + '=' + answer)
    except AttributeError:
        print(equation)


def predict_with_box(classifier, image):
    clf = joblib.load(classifier)

    im = cv2.imread(image)

    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)

    ret, im_th = cv2.threshold(im_gray, 110, 255, cv2.THRESH_BINARY_INV)

    _, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_TREE,
                                     cv2.CHAIN_APPROX_NONE)

    rects = [cv2.boundingRect(ctr) for ctr in ctrs]

    for rect in rects:
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        if not roi.size:
            continue
        roi = cv2.resize(roi, (45, 45), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        roi_hog_fd = hog(roi, orientations=6, pixels_per_cell=(15, 15),
                         cells_per_block=(1, 1), visualise=False)
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)

    cv2.imshow('', im)
    cv2.waitKey()


def main(classifier, image):
    convert_to_equation(predict(classifier, *get_contours(image)))
    predict_with_box(classifier, image)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--classifier', required=True,
                    help='path to classifier')
    ap.add_argument('-i', '--image', required=True,
                    help='path to image')
    args = vars(ap.parse_args())

    main(**args)

