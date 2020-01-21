from keras.layers import Conv2D, MaxPooling2D, Input, Reshape, BatchNormalization, LeakyReLU, Concatenate, Permute
from keras import Model
from keras.models import load_model
import numpy as np


class DrivingObjectDetector:
    def __init__(self):

        self.classes = ['person',
                        'bicycle', 'skateboard',
                        'car', 'truck', 'motorcycle', 'bus', 'train',
                        'traffic light', 'stop sign', 'parking meter',
                        'sports ball', 'animal']

        self.anchors = np.array([
            0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828
        ]).reshape(5, 2)

        self.image_size = 608
        self.cell_number = 19
        self.cell_size = 32
        self.output_shape = (self.cell_number, self.cell_number, len(self.anchors), 4 + 1 + len(self.classes))

        self.model = self.build_model()
        print(self.model.summary())

    def import_weights(self, read_model_path, first=0, last=None):
        read_model = load_model(read_model_path)

        if last is None:
            last = len(self.model.layers)

        for i in range(first, last):

            self.model.layers[i].set_weights(read_model.layers[i].get_weights())

            w = np.array(self.model.layers[i].get_weights())
            if np.array_equal(np.array(read_model.layers[i].get_weights()), w):
                print('Imported weights of shape {} for layer index {}'.format(w.shape, i))
        read_model = None  # unload read model form GPU memory

    def build_model(self):
        model_in = Input((self.image_size, self.image_size, 3))

        model = conv_composite(model_in, 32, 3, train=False)
        model = MaxPooling2D(2, padding='valid')(model)

        model = conv_composite(model, 64, 3, train=False)
        model = MaxPooling2D(2, padding='valid')(model)

        model = network_block(model, [(128, 3), (64, 1), (128, 3)], train=False)
        model = MaxPooling2D(2, padding='valid')(model)

        model = network_block(model, [(256, 3), (128, 1), (256, 3)], train=False)
        model = MaxPooling2D(2, padding='valid')(model)

        model = network_block(model, [(512, 3), (256, 1), (512, 3), (256, 1), (512, 3)], train=False)
        skip = model
        model = MaxPooling2D(2, padding='valid')(model)

        model = network_block(model, [(1024, 3), (512, 1), (1024, 3), (512, 1), (1024, 3)], train=False)

        model = conv_composite(model, 1024, 3)
        model = conv_composite(model, 1024, 3)

        skip = conv_composite(skip, 64, 1)
        model = Concatenate()([reorganize(skip, 2), model])

        model = conv_composite(model, 1024, 3)

        outputs_number = len(self.anchors) * (5 + len(self.classes))
        model = Conv2D(outputs_number, (1, 1), padding='same', activation='linear')(model)

        model_out = Reshape(self.output_shape)(model)

        return Model(inputs=model_in, outputs=model_out)


def conv_composite(tensor, filters, dim, strides=1, train=True):
    tensor = Conv2D(filters, (dim, dim), strides=strides, padding='same', use_bias=False, trainable=train)(tensor)
    tensor = BatchNormalization()(tensor)
    tensor = LeakyReLU(alpha=0.1)(tensor)
    return tensor


def network_block(tensor, dims, train=True):
    for d in dims:
        tensor = conv_composite(tensor, *d, train=train)
    return tensor


# TODO: Retrain the detection section of the model to support residual layer connectivity and remove the reorganise function
def reorganize(input_tensor, stride):
    _, h, w, c = input_tensor.get_shape().as_list()

    channel_first = Permute((3, 1, 2))(input_tensor)

    reshape_tensor = Reshape((c // (stride ** 2), h, stride, w, stride))(channel_first)
    permute_tensor = Permute((3, 5, 1, 2, 4))(reshape_tensor)
    target_tensor = Reshape((-1, h // stride, w // stride))(permute_tensor)

    channel_last = Permute((2, 3, 1))(target_tensor)
    return Reshape((h // stride, w // stride, -1))(channel_last)


