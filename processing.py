import numpy as np


def process_training_predictions(predictions, anchors, network_output_shape):
    """
    Converts predictions list to predictions vector compatible with the loss function.
    Predictions vector cells are populated with respect to bounding box location (cell indices)
    and shape (anchor indices)
    :param predictions: predictions list for image, corresponds to one item in batch (from batcher)
    :param anchors: anchor values from detector
    :param network_output_shape: shape of final layer in detector network
    :return: y_true vector to be used in training
    """

    boxes = predictions[:, :4] / 32  # divide by cell size - this comes from image_width / nr of cells
    # TODO: cell size needs to be adaptable to different nr of cells, network_output_shape fixed (19,19,?,?) !!!

    # convert top-left x-y values to center x-y
    x1, y1, x2, y2 = np.split(boxes, 4, axis=1)
    w = x2 - x1
    h = y2 - y1
    c_x = x1 + w / 2
    c_y = y1 + h / 2
    boxes = np.concatenate([c_x, c_y, w, h], axis=-1)

    # create classes one-hot encoding
    labels = np.zeros(
        shape=(predictions.shape[0], network_output_shape[3] - 5))
    for i, p in enumerate(predictions[:, 4]):
        labels[i][int(p)] = 1

    # find the best anchors for predictions
    w_boxes, h_boxes = boxes[:, 2], boxes[:, 3]
    w_anchors, h_anchors = anchors[:, 0], anchors[:, 1]
    horizontal_overlap = np.minimum(w_boxes[:, None], w_anchors)
    vertical_overlap = np.minimum(h_boxes[:, None], h_anchors)
    intersection = horizontal_overlap * vertical_overlap
    union = (w_boxes * h_boxes)[:, None] + (w_anchors * h_anchors) - intersection
    iou = intersection / union
    best_anchor_indices = np.argmax(iou, axis=1)

    # cells to be populated with predictions
    responsible_grid_cells = np.floor(boxes).astype(np.uint32)[:, :2]

    # all values for predictions vector
    values = np.concatenate((boxes, np.ones((len(boxes), 1)), labels), axis=1)

    # create responsible cell indices
    x_cell_indices, y_cell_indices = np.split(responsible_grid_cells, 2, axis=1)
    y_cell_indices = y_cell_indices.ravel()
    x_cell_indices = x_cell_indices.ravel()

    # create empty vector and populate with values
    y_true = np.zeros(network_output_shape)
    y_true[y_cell_indices, x_cell_indices, best_anchor_indices] = values

    return y_true


def process_training_batch(batch, anchors, network_output_shape):
    """
    Applies process_training_predictions on all items in batch.
    Converts all images in batch from int to 0-1 float values.
    :param batch: Batch to be processed, from batcher
    :param anchors: anchor values from detector
    :param network_output_shape: shape of final layer in detector network
    :return: Batch of image, predictions pairs as array for training
    """

    images, predictions = batch
    images = images / 255.
    predictions_batch = []
    for pred in predictions:
        predictions_batch.append(process_training_predictions(pred, anchors, network_output_shape))
    return images, np.asarray(predictions_batch)


def process_output(output,
                   anchors,
                   confidence_threshold=0.5,
                   max_suppression_thresh=0.5,
                   max_suppression=True,
                   cell_size=32):

    """
    Converts detector network raw output to a list of predictions.
    :param output: detector network output for one item in batch
    :param anchors: anchor values from detector
    :param confidence_threshold: minimum confidence value for detection
    :param max_suppression_thresh: non max suppression threshold
    :param max_suppression: bool, use non max suppression
    :param cell_size: detection cell size
    :return: predictions list
    """

    pw, ph = anchors[:, 0], anchors[:, 1]
    cell_inds = np.arange(output.shape[1])

    tx = output[..., 0]
    ty = output[..., 1]
    tw = output[..., 2]
    th = output[..., 3]
    to = output[..., 4]

    sftmx = softmax(output[..., 5:])
    predicted_labels = np.argmax(sftmx, axis=-1)
    class_confidences = np.max(sftmx, axis=-1)

    # '''
    # Running code for model predictions
    bx = logistic(tx) + cell_inds[None, :, None]
    by = logistic(ty) + cell_inds[:, None, None]
    bw = pw * np.exp(tw) / 2
    bh = ph * np.exp(th) / 2
    object_confidences = logistic(to)

    left = bx - bw
    right = bx + bw
    top = by - bh
    bottom = by + bh
    # '''

    '''
    # Test code for training data processing
    left = tx - tw / 2
    right = tx + tw / 2
    top = ty - th / 2
    bottom = ty + th / 2
    object_confidences = to
    #'''

    boxes = np.stack((left, top, right, bottom), axis=-1) * cell_size
    final_confidence = class_confidences * object_confidences

    boxes = boxes[final_confidence > confidence_threshold].reshape(-1, 4).astype(np.int32)
    labels = predicted_labels[final_confidence > confidence_threshold].reshape(-1, 1).astype(np.int32)

    if max_suppression and len(boxes) > 0:
        boxes, labels = non_max_suppression(boxes, labels, max_suppression_thresh)

    return np.concatenate((boxes, labels), axis=-1)


def process_output_batch(output_batch,
                         anchors,
                         confidence_threshold=0.5,
                         max_suppression_thresh=0.5,
                         max_suppression=True,
                         cell_size=32):

    """
    Applies process_output on all items in batch.

    :param output_batch: detector network output
    :param anchors: anchor values from detector
    :param confidence_threshold: minimum confidence value for detection
    :param max_suppression_thresh: non max suppression threshold
    :param max_suppression: bool, use non max suppression
    :param cell_size: detection cell size
    :return:batch of predictions lists
    """

    predictions_batch = []
    for output in output_batch:
        predictions = process_output(output,
                                     anchors,
                                     confidence_threshold,
                                     max_suppression_thresh,
                                     max_suppression,
                                     cell_size)
        predictions_batch.append(predictions)

    return predictions_batch


# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression(boxes, labels, overlap_threshold):

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        i = idxs[-1]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:-1]])
        yy1 = np.maximum(y1[i], y1[idxs[:-1]])
        xx2 = np.minimum(x2[i], x2[idxs[:-1]])
        yy2 = np.minimum(y2[i], y2[idxs[:-1]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:-1]]

        # delete all indexes from the index list that have
        idxs = (idxs[:-1])[overlap < overlap_threshold]

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick], labels[pick]


"""
Logistic and Softmax activations for processing the model output (final stages) at CPU level
There are parallel versions of these running in the loss function. This necessary in order to 
maintain Keras compatibility (no custom Keras layers required, minimal use of tensorflow backend)
This also makes porting to Tensorflow 2.0 more streamlined
"""


def logistic(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    exp_out = np.exp(x - np.max(x, axis=-1)[..., None])
    return exp_out / np.sum(exp_out, axis=-1)[..., None]



