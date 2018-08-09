import argparse
import os
import numpy as np
import pandas as pd
from PIL import Image
from skimage.feature import hog
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

image_types = np.arange(0, 10).astype(str).tolist() \
              + ['+', '-', 'div', 'times', 'int', 'y', 'd']
image_types_to_labels = {image_type: label for label, image_type
                         in enumerate(image_types)}
labels_to_image_type = {label: image_type for label, image_type
                        in enumerate(np.arange(0, 10).astype(str).tolist()
                                     + ['+', '-', '-', '*', 'integrate ',
                                        'y', 'd'])}


def predict(clf_path, data_path):
    clf_path = r'C:\Users\phamb1\Desktop\Summer 2018\Embedded AI\Project\handwritten_calculator'
    clf = joblib.load(os.path.join(clf_path, 'integrate_cls.pkl'))
    data_path = r'C:\Users\phamb1\Desktop\Summer 2018\Embedded AI\Project\data'

    features, labels = [], []

    for folder in os.listdir(data_path):
        if folder in image_types:
            for image_path in os.listdir(os.path.join(data_path, folder))[2000:2500]:
                image = Image.open(os.path.join(data_path, folder, image_path))
                image.convert('L')
                features.append(np.array([
                    point for point in image.getdata()]))
                labels.append(np.array([image_types_to_labels[folder]]))

    list_hog_fd = [hog(feature.reshape((45, 45)), orientations=6,
                       pixels_per_cell=(15, 15), cells_per_block=(1, 1),
                       visualise=False)
                   for feature in features]

    prediction = clf.predict(np.array(list_hog_fd))
    print(accuracy_score(prediction, np.ravel(labels)))


def create_classifier(data_path):	
	data_path = r'C:\Users\phamb1\Desktop\Summer 2018\Embedded AI\Project\data'

    features, labels = [], []

    for folder in os.listdir(data_path):
        if folder in image_types:
            for image_path in os.listdir(os.path.join(data_path, folder))[:2000]:
                image = Image.open(os.path.join(data_path, folder, image_path))
                image.convert('L')
                features.append(np.array([
                    point for point in image.getdata()]))
                labels.append(np.array([image_types_to_labels[folder]]))

    # np.savetxt('data_set.txt', np.vstack(flattened), fmt='%.0f', delimiter=',')
    # np.savetxt('labels.txt', np.vstack(labels), fmt='%.0f', delimiter=',')
    #
    # path = r'C:\Users\phamb1\Desktop\Summer 2018\Embedded AI\Project\handwritten_calculator'
    # features = np.loadtxt(os.path.join(path, 'data_set.txt'), delimiter=',')
    # labels = np.loadtxt(os.path.join(path, 'labels.txt'), delimiter=',')

    list_hog_fd = [hog(feature.reshape((45, 45)), orientations=6,
                       pixels_per_cell=(15, 15), cells_per_block=(1, 1),
                       visualise=False)
                   for feature in features]
    hog_features = np.array(list_hog_fd, 'float64')
    # clf = SVC().fit(hog_features, np.ravel(labels))
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(18, ),
						max_iter=500, random_state=1).fit(hog_features, np.ravel(labels))
    joblib.dump(clf, 'integrate_cls.pkl', compress=3)


def main(args):
    create_classifier(args['data'])
	predict(args['clf'], args['data'])


if __name__ == '__main__':
ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--classifier', required=True,
                    help='path to classifier')
    ap.add_argument('-i', '--image', required=True,
                    help='path to data')
    args = vars(ap.parse_args())
	
    main()
