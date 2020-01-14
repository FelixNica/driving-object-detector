import json
import os
import cv2 as cv
import time
import threading
import numpy as np

"""
Generate batches of images with their corresponding predictions in list format from pre-processed data.
Handles fetch-from-storage using treading during GPU compute time to cut down on total training time.
"""
# TODO: add NetworkBatcher for fetching data from network location (Performance strategy should apply)


class LocalBatcher:

    """
    Generates batches form locally stored data.
    Threading gives maximum performance gain when the data is very big in size and stored on disk drives.
    Minimum performance gain using SSD but limited by storage capacity.

    Restarts bach generation at end of data.
    Can restart bach generation after a set number of batches with "restart_after" parameter.
    """

    def __init__(self, data_path, batch_size, restart_after=None):

        self.data_path = data_path
        self.batch_size = batch_size
        self.restart = restart_after
        self.name = 'LocalBatcher-' + self.data_path + ':'

        if os.path.isfile(self.data_path + 'annotations.json') is False:
            print(self.name, 'No annotations file present for path')

        with open(self.data_path + 'annotations.json') as json_file:
            self.annotations_data = json.load(json_file)
        print(self.name, 'Data annotations loaded from: {}'.format(self.data_path + 'annotations.json'))

        self.item_flow = self.item_generator()
        self.batch_flow = self.batch_generator()

        self.batch_buffer = None
        self.safe_flag = True
        self.batch_buffer = self.get_batch()
        # Batch is stored in buffer until GPU is ready to load.

    def item_generator(self):

        for image_name in self.annotations_data.keys():
            if not os.path.exists(self.data_path + image_name):
                print(self.name, 'Image {} not found, Data item skipped!'.format(self.data_path + image_name))
                continue

            image = cv.imread(self.data_path + image_name)

            predictions = []
            for pred in self.annotations_data[image_name]:
                prediction = (pred[0], pred[1], pred[2], pred[3], pred[4])
                predictions.append(prediction)

            yield image, np.asarray(predictions)

    def batch_generator(self):
        counter = 0
        while True:

            X = []
            Y = []

            if counter == self.restart:
                self.item_flow = self.item_generator()
                print(self.name, 'Restarted'.format(self.data_path))
                counter = 0
            counter += 1

            for i in range(self.batch_size):
                try:
                    x, y = next(self.item_flow)
                except UnboundLocalError:
                    self.item_flow = self.item_generator()
                    print(self.name, 'Restarted'.format(self.data_path))
                    x, y = next(self.item_flow)
                except StopIteration:
                    self.item_flow = self.item_generator()
                    print(self.name, 'Restarted'.format(self.data_path))
                    x, y = next(self.item_flow)

                X.append(x)
                Y.append(y)

            yield np.asarray(X), np.asarray(Y)

    def get_batch(self):
        """
        Launches a parallel bach compute thread to load new data after the current buffer is handed to the GPU.
        :return: current batch to be handed to GPU
        """
        wait_start = time.time()
        while self.safe_flag is False:
            time.sleep(0.1)
        wait_end = time.time() - wait_start
        if wait_end > 0.2:
            print(self.name, "Waited for bach compute {} seconds".format(self.data_path, wait_end))
        threading.Thread(target=self._compute_batch).start()
        return self.batch_buffer

    def _compute_batch(self):
        self.safe_flag = False
        self.batch_buffer = next(self.batch_flow)
        self.safe_flag = True


