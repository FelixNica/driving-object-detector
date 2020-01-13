"""
CREDIT: Guillaume Ramé - London, UK
https://github.com/guigzzz

After spending two months developing my own loss function I was not
satisfied with the approach so I adapted Guillaume Ramé's excellent
version to my project. No license was attached to his code but please
reach out to him if you intend to distribute this.

Original paper by Joseph Redmon:
https://arxiv.org/pdf/1506.02640.pdf

Formally we define confidence as Pr(Object) ∗ IOU^truth_pred. 
If no object exists in that cell, the confidence scores should be
zero. Otherwise we want the confidence score to equal the
intersection over union (IOU) between the predicted box
and the ground truth.

Each bounding box consists of 5 predictions: x, y, w, h,
and confidence. The (x, y) coordinates represent the center
of the box relative to the bounds of the grid cell. The width
and height are predicted relative to the whole image. Finally
the confidence prediction represents the IOU between the
predicted box and any ground truth box. (specific to v1)
"""

import tensorflow as tf


def get_function(anchors):
    """
    Wrapper function, used to pass the loss function with anchor
    values to Tensorflow. A reference to loss_function gets saved
    as a custom object in the model training configuration.
    This will need to be passed at every model compile.
    :param anchors: anchor values from detector
    :return: loss_function
    """

    def loss_function(y_true, y_pred):
        n_cells = y_pred.get_shape().as_list()[1]
        y_true = tf.reshape(y_true, tf.shape(y_pred), name='y_true')
        y_pred = tf.identity(y_pred, name='y_pred')

        # PROCESS PREDICTIONS FROM MODEL #
        # Final sigmoid activation for x-y values
        predicted_xy = tf.nn.sigmoid(y_pred[..., :2])

        # convert x-y form "respective to CELL" to "respective to IMAGE"
        cell_indexes = tf.range(n_cells, dtype=tf.float32)
        predicted_xy = tf.stack((
            predicted_xy[..., 0] + tf.reshape(cell_indexes, [1, -1, 1]),
            predicted_xy[..., 1] + tf.reshape(cell_indexes, [-1, 1, 1])
        ), axis=-1)

        # compute bb width and height
        predicted_wh = anchors * tf.exp(y_pred[..., 2:4])

        # compute predicted bb center and width
        predicted_min = predicted_xy - predicted_wh / 2
        predicted_max = predicted_xy + predicted_wh / 2

        predicted_objectedness = tf.nn.sigmoid(y_pred[..., 4])
        predicted_logits = tf.nn.softmax(y_pred[..., 5:])

        # PROCESS GROUND TRUTH #
        true_xy = y_true[..., :2]
        true_wh = y_true[..., 2:4]
        true_logits = y_true[..., 5:]

        true_min = true_xy - true_wh / 2
        true_max = true_xy + true_wh / 2

        # compute iou between ground truth and predicted (used for objectedness) #
        intersect_mins = tf.maximum(predicted_min, true_min)
        intersect_maxes = tf.minimum(predicted_max, true_max)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = predicted_wh[..., 0] * predicted_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = intersect_areas / union_areas

        # COMPUTE LOSS TERMS #
        responsibility_selector = y_true[..., 4]

        xy_diff = tf.square(true_xy - predicted_xy) * responsibility_selector[..., None]
        xy_loss = tf.reduce_sum(xy_diff, axis=[1, 2, 3, 4])

        wh_diff = tf.square(tf.sqrt(true_wh) - tf.sqrt(predicted_wh)) * responsibility_selector[..., None]
        wh_loss = tf.reduce_sum(wh_diff, axis=[1, 2, 3, 4])

        obj_diff = tf.square(iou_scores - predicted_objectedness) * responsibility_selector
        obj_loss = tf.reduce_sum(obj_diff, axis=[1, 2, 3])

        best_iou = tf.reduce_max(iou_scores, axis=-1)
        no_obj_diff = tf.square(0 - predicted_objectedness) * tf.to_float(best_iou < 0.6)[..., None] * (
                    1 - responsibility_selector)
        no_obj_loss = tf.reduce_sum(no_obj_diff, axis=[1, 2, 3])

        clf_diff = tf.square(true_logits - predicted_logits) * responsibility_selector[..., None]
        clf_loss = tf.reduce_sum(clf_diff, axis=[1, 2, 3, 4])

        # loss scaling factors #
        object_coord_scale = 5
        object_conf_scale = 2
        noobject_conf_scale = 1
        object_class_scale = 1

        loss_value = object_coord_scale * (xy_loss + wh_loss) + \
                     object_conf_scale * obj_loss + noobject_conf_scale * no_obj_loss + \
                     object_class_scale * clf_loss
        return loss_value

    return loss_function


