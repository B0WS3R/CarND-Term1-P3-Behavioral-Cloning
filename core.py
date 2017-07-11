########################################################################################################################
# Udacity Self-Driving Car Nanodegree
# Term1-P3 Behavioral cloning
########################################################################################################################

import csv
import cv2
import numpy as np
from sklearn.utils import shuffle

########################################################################################################################
# Principal function: load and returns data/labels
########################################################################################################################

def generate_data():
    # my behavioral cloning scenerarios
    scenarios = [
        't1-right',
        't1-normal',
        't1-left',
        't1-backward',
        't1-corrective',
        ##'udacity-data',
    ]

    image_load_data = []
    for data_scenario in scenarios:
        print('Getting csv lines for scenario:', data_scenario)
        with open('../training-data/{}/driving_log.csv'.format(data_scenario)) as f:
            reader = csv.reader(f)
            for csv_line in reader:
                image_load_data.append((csv_line, data_scenario))

    image_load_data = shuffle(image_load_data)
    images = []
    angles = []
    for line, folder in image_load_data:
        # I only used the center camera image
        center_img_path = line[0]
        file_name = center_img_path.split('/')[-1]
        image = cv2.imread('../training-data/{}/IMG/{}'.format(folder, file_name))
        if image is not None:
            images.append(image)
            # flip the image
            images.append(np.fliplr(image))
            steering_angle = float(line[3])
            angles.append(steering_angle)
            # flip the steering angle
            angles.append(-steering_angle)

    return np.array(images), np.array(angles)