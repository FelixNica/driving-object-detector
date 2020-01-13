import os
import csv
import time
import utils
import numpy as np
import json
import processing
import keras.backend as K


def init_experiment(path):
    """
    Local experiment initialisation.
    :param path: path of experiment folder
    :return: experiment configuration dict
    """

    if os.path.exists(path) is False:
        os.mkdir(path)
        print('Experiment directory "{}" created'.format(path))

    if os.path.exists(path + '/training_samples') is False:
        os.mkdir(path + '/training_samples')

    if os.path.exists(path + '/test_samples') is False:
        os.mkdir(path + '/test_samples')

    if os.path.exists(path + '/model_states') is False:
        os.mkdir(path + '/model_states')

    if os.path.isfile(path + '/experiment_configuration.json') is False:
        config_dict = {'average_every': 100,
                       'validate_every': 20,
                       'update_plots_every': 100,
                       'save_state_every': 1000,
                       'save_sample_every': 200,
                       'total_epochs': 0,
                       'total_time': 0}

        with open(path + '/experiment_configuration.json', 'w') as cfg:
            json.dump(config_dict, cfg)
        print('Experiment configuration initialised form default')
        return config_dict

    else:
        with open(path + '/experiment_configuration.json') as cfg:
            config_dict = json.load(cfg)
        print('Experiment configuration loaded from file')
        return config_dict


def start_session(experiment_path,
                  detector,
                  data_flow_train,
                  data_flow_test,
                  epochs,
                  lr_scheduler=None,
                  test_only=False):

    """
    Training & Testing / Testing only session
    :param experiment_path: path of experiment folder
    :param detector: detector object
    :param data_flow_train: training data flow
    :param data_flow_test: testing data flow
    :param epochs: Number of epochs to train/test
    :param lr_scheduler: function, dynamic learning rate hyper-parameter
    :param test_only: If true session will just test the model without optimising
    """

    config = init_experiment(experiment_path)
    metrics_keys = ['train_loss', 'val_loss']

    if os.path.isfile(experiment_path + '/metrics_full.csv') is False:
        with open(experiment_path + '/metrics_full.csv', 'a', newline='') as f:
            csv.writer(f).writerow(metrics_keys)

    if os.path.isfile(experiment_path + '/metrics_avg.csv') is False:
        with open(experiment_path + '/metrics_avg.csv', 'a', newline='') as f:
            csv.writer(f).writerow(metrics_keys)

    metrics_buffer = []
    session_start_time = time.time()

    train_loss = 0
    val_loss = 0

    for epoch in range(config['total_epochs'], config['total_epochs']+epochs):

        if test_only is False:
            imgs, preds = data_flow_train.get_batch()
            x_train, y_train = processing.process_training_batch((imgs, preds),
                                                                 detector.anchors,
                                                                 detector.output_shape)

            if lr_scheduler is not None:
                lr = lr_scheduler(epoch)
                K.set_value(detector.model.optimizer.lr, lr)

            train_loss = detector.model.train_on_batch(x_train, y_train)

            if epoch % config['save_sample_every'] is 0:
                z_train = detector.model.predict(x_train, batch_size=x_train.shape[0])
                z_train = processing.process_output_batch(z_train, detector.anchors,
                                                          confidence_threshold=0.05, max_suppression=False)
                spl_name = experiment_path+'/training_samples/train_e-'+str(epoch)
                utils.sample_batch(imgs, z_train, detector.classes, spl_name)

            if epoch % config['save_state_every'] is 0 and epoch is not 0:
                detector.model.save(experiment_path + '/model_states/state_e-'+str(epoch))

        if test_only is True or epoch % config['validate_every'] is 0:
            imgs, preds = data_flow_test.get_batch()
            x_test, y_test = processing.process_training_batch((imgs, preds),
                                                               detector.anchors,
                                                               detector.output_shape)
            val_loss = detector.model.evaluate(x_test, y_test, y_test.shape[0], verbose=0)

            if epoch % config['save_sample_every'] is 0:
                z_test = detector.model.predict(x_test, batch_size=x_test.shape[0])
                z_test = processing.process_output_batch(z_test, detector.anchors)
                spl_name = experiment_path + '/test_samples/test_e-' + str(epoch)
                utils.sample_batch(imgs, z_test, detector.classes, spl_name)

        metrics_list = [train_loss, val_loss]
        metrics_buffer.append(metrics_list)
        if epoch % config['average_every'] is 0:

            metrics_avg = np.mean(metrics_buffer, axis=0)
            print(80*'_')
            print('session_epoch:        ' + str(epoch-config['total_epochs']))
            print('experiment_epoch:     ' + str(epoch))
            print(' ')
            for i in range(len(metrics_list)):
                print(metrics_keys[i])
                print(' on_epoch:            ' + str(metrics_list[i]))
                print(' on_average:          ' + str(metrics_avg[i]))

            with open(experiment_path + '/metrics_full.csv', 'a', newline='') as f:
                for entry in metrics_buffer:
                    csv.writer(f).writerow(entry)
            with open(experiment_path + '/metrics_avg.csv', 'a', newline='') as f:
                csv.writer(f).writerow(metrics_avg)

            metrics_buffer = []

        if epoch % config['update_plots_every'] is 0 and epoch is not 0:
            utils.plot(experiment_path + '/metrics_avg.csv', experiment_path + '/plot')

    if test_only is False:
        detector.model.save(experiment_path + '/model_states/state_final')

    config['total_epochs'] = config['total_epochs']+epochs
    session_time = time.time() - session_start_time
    config['total_time'] = config['total_time'] + session_time
    with open(experiment_path + '/experiment_configuration.json', 'w') as cfg:
        json.dump(config, cfg)

    print(80*'_')
    print('session_time:')
    print(' hours:               ' + str(session_time / 3600))
    print(' minutes:             ' + str(session_time / 60))
    print('total_time:')
    print(' hours:               ' + str(config['total_time'] / 3600))
    print(' minutes:             ' + str(config['total_time'] / 60))
    print(80*'=')


