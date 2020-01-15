import processing
import utils
import glob
import cv2 as cv
import numpy as np
import os


def deploy(detector, input_path, output_path, batch_size, conf_thresh=0.5, max_supp_thresh=0.5):
    image_list = glob.glob(input_path + '/*.jpg')

    if os.path.exists(output_path) is False:
        os.mkdir(output_path)

    for i in range(int(len(image_list)/batch_size)+1):
        image_batch = []
        for image_name in image_list[i*batch_size:(i+1)*batch_size]:
            image = cv.imread(image_name)
            image = cv.resize(image, (608, 608))
            image_batch.append(image)

        if len(image_batch) is 0:
            continue
        image_batch = np.asarray(image_batch)
        print(np.asarray(image_batch).shape)

        pr_img = image_batch / 255.

        pred_batch = detector.model.predict(pr_img, batch_size=pr_img.shape[0])
        pred_batch = processing.process_output_batch(pred_batch, detector.anchors,
                                                     conf_thresh=conf_thresh, max_supp_thresh=max_supp_thresh)
        spl_name = output_path + '/sample_batch-' + str(i)
        utils.sample_batch(image_batch, pred_batch, detector.classes, spl_name)


