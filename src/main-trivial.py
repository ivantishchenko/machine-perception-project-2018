#!/usr/bin/env python3
"""Main script for training a model for hand joint recognition."""
import argparse
import coloredlogs
import tensorflow as tf

# HYPER PARAMETER TUNINGS HERE
BATCHSIZE = 32
EPOCHS = 25

if __name__ == '__main__':

    # Set global log level
    parser = argparse.ArgumentParser(description='Train a 2D joint estimation model.')
    parser.add_argument('-v', type=str, help='logging level', default='info',
                        choices=['debug', 'info', 'warning', 'error', 'critical'])
    args = parser.parse_args()
    coloredlogs.install(
        datefmt='%d/%m %H:%M',
        fmt='%(asctime)s %(levelname)s %(message)s',
        level=args.v.upper(),
    )

    # Initialize Tensorflow session
    tf.logging.set_verbosity(tf.logging.INFO)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:

        # Declare some parameters
        batch_size = BATCHSIZE

        # Define model
        from datasources import HDF5Source
        from models import TrivialNet
        model = TrivialNet(
            # Note: The same session must be used for the model and the data sources.
            session,

            learning_schedule=[
                {
                    'loss_terms_to_optimize': {
                        'kp_loss_mse': ['keypoints', 'flatten', 'loss_calculation'],
                    },
                    'metrics': ['kp_loss_mse', 'kp_accuracy'],
                    'learning_rate': 1e-4,
                },
            ],

            test_losses_or_metrics=['kp_loss_mse', 'kp_accuracy'],

            # Data sources for training and testing.
            train_data={
                'real': HDF5Source(
                    session,
                    batch_size,
                    hdf_path='../datasets/training.h5',
                    keys_to_use=['train'],
                    min_after_dequeue=2000,
                ),
            },

            # test_data={
            #     'real': HDF5Source(
            #         session,
            #         batch_size,
            #         hdf_path='../datasets/dataset.h5',
            #         keys_to_use=['validate'],
            #         testing=True,
            #     ),
            # },
        )

        # Train this model for a set number of epochs
        model.train(
            num_epochs=EPOCHS,
        )

        # Test for Kaggle submission
        model.evaluate_for_kaggle(
            HDF5Source(
                session,
                batch_size,
                hdf_path='../datasets/testing.h5',
                keys_to_use=['test'],
                testing=True,
            )
        )
