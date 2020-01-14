import matplotlib.pyplot as plt
import csv
import numpy as np
import os
import cv2 as cv


# annotate image with predictions
def annotate(image, predictions, class_names):
    image = image.copy()
    for (left, top, right, bottom, label) in predictions:
        cv.rectangle(image, (left, top),
                     (right, bottom), color=(255, 0, 0), thickness=2)
        cv.putText(
            image, class_names[label], (left, top-10),
            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
            thickness=2
        )
    return image


# annotate batch and save samples
def sample_batch(image_batch, predictions_batch, class_names, name):
    for i in range(len(image_batch)):
        annotated_image = annotate(image_batch[i], predictions_batch[i], class_names)
        cv.imwrite(name+'_'+str(i)+'.jpg', annotated_image)


# plot graph from metrics file
def plot(input_file, output_path):
    plt.yscale("log")
    plt_table = []
    plt_keys = []
    with open(input_file) as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i is 0:
                plt_keys = row
            else:
                float_row = []
                for number in row:
                    if number is 0:
                        number = 0.000001
                    float_row.append(float(number))
                plt_table.append(float_row)
    plt_table = np.array(plt_table)

    for i, key in enumerate(plt_keys):
        plt.plot(plt_table[:, i], label=key)
    plt.legend()
    if os.path.isfile(output_path + '.png') is True:
        os.remove(output_path + '.png')
    plt.savefig(output_path, dpi=1000)
    plt.close()


