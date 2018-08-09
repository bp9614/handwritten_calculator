import argparse
import os
import io
import cv2
import numpy as np
import pytesseract
from PIL import Image


def to_greyscale(image, preprocess='thresh'):
    gray = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)

    if preprocess not in ['thresh', 'blur']:
        raise ValueError('Only thresh or blur can be used as a preprocess.')

    if preprocess.lower() == 'thresh':
        gray = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    if preprocess.lower() == 'blur':
        gray = cv2.medianBlur(gray, 3)

    in_memory = io.BytesIO(cv2.imencode('.png', gray)[1].tostring())
    in_memory.seek(0)
    return in_memory


def extract_text(image):
    return pytesseract.image_to_string(Image.open(image))


def main(args):
    grayscale_image = to_greyscale(**args)
    text = extract_text(grayscale_image)
    print(text)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image', required=True,
                    help='path to input image to be OCR\'d')
    ap.add_argument('-p', '--preprocess', type=str, default='thresh',
                    help='type of preprocessing to be done')
    args = vars(ap.parse_args())
    main(args)
