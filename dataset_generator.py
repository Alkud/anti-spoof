import os
import numpy as np
import pandas as pd
import random
from glob import glob

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical


class DatasetGenerator:
    def __init__(self, label_set, train_input_path, test_input_path, bit_depth):
        self.label_set = label_set
        self.train_data_directory = train_input_path
        self.test_data_directory = test_input_path
        self.npy_bit_depth = bit_depth
        self.full_data_frame = pd.DataFrame(
            [('0', 'human', 'human_00000.npy')], columns=['label', 'label_id', 'npy_path']
        )
        self.train_data_frame = pd.DataFrame(
            [('0', 'human', 'human_00000.npy')], columns=['label', 'label_id', 'npy_path']
        )
        self.validation_data_frame = pd.DataFrame(
            [('0', 'human', 'human_00000.npy')], columns=['label', 'label_id', 'npy_path']
        )
        self.test_data_frame = pd.DataFrame(
            [('0', 'human', 'human_00000.npy')], columns=['label', 'label_id', 'npy_path']
        )
        self.wild_test_data_frame = pd.DataFrame(
            ['sample_00000.npy'], columns=['npy_path']
        )

    # Covert string to numerical classes
    def text_to_labels(self, text):
        return self.label_set.index(text)

    # Reverse translation of numerical classes back to characters
    def labels_to_text(self, labels):
        return self.label_set[labels]

    def load_data(self):
        # get all .npy files inside data_path directory
        npy_files = glob(os.path.join(self.train_data_directory + '/npy/', '*/*.npy'))
        # loop over files to get gd-grams
        train_data = []
        for full_npy_path in npy_files:
            directory_name, full_file_name = os.path.split(full_npy_path)
            file_name, ext = os.path.splitext(full_file_name)
            label, name = file_name.split('_')
            if label in self.label_set:
                label_id = self.text_to_labels(label)
                sample = (label, label_id, full_npy_path)
                train_data.append(sample)

        # Data Frames with samples' labels, label_id's and paths
        train_data_frame = pd.DataFrame(train_data, columns=['label', 'label_id', 'npy_path'])
        self.full_data_frame = train_data_frame

        #wild test data frame
        npy_files = glob(os.path.join(self.test_data_directory + '/npy/', '*.npy'))
        test_data = []
        for full_npy_path in npy_files:
            test_data.append(full_npy_path)
        self.wild_test_data_frame = pd.DataFrame(test_data, columns=['npy_path'])
        return self.full_data_frame

    def apply_train_test_split(self, test_size, random_state):
        self.train_data_frame, self.test_data_frame = \
            train_test_split(self.full_data_frame,
                             test_size=test_size,
                             random_state=random_state)

    def apply_train_validation_split(self, validation_size, random_state):
        self.train_data_frame, self.validation_data_frame = \
            train_test_split(self.full_data_frame,
                             test_size=validation_size,
                             random_state=random_state)

    def read_npy_file(self, file_path):
        # read np.ndarray from file
        gd_array = np.load(file_path).astype(float)
        gd_array /= 2**self.npy_bit_depth - 1
        return np.expand_dims(gd_array, axis=2)

    def generator(self, batch_size, mode):
        while True:
            if mode == 'train':
                data_frame = self.train_data_frame
                sample_indices = random.sample(range(data_frame.shape[0]), data_frame.shape[0])
            elif mode == 'validation':
                data_frame = self.validation_data_frame
                sample_indices = list(range(data_frame.shape[0]))
            elif mode == 'test':
                data_frame = self.test_data_frame
                sample_indices = list(range(data_frame.shape[0]))
            elif mode == 'wild_test':
                data_frame = self.wild_test_data_frame
                sample_indices = list(range(data_frame.shape[0]))
            else:
                raise ValueError('The mode should be either train, val or test.')

            # create batches (for training data the batches are randomly permuted)
            for start in range(0, len(sample_indices), batch_size):
                samples_batch = []
                if mode != 'test' and mode != 'wild_test':
                    labels_batch = []
                end = min(start + batch_size, len(sample_indices))
                batch_indices = sample_indices[start:end]
                for index in batch_indices:
                    samples_batch.append(self.read_npy_file(data_frame['npy_path'].values[index]))
                    if mode != 'test' and mode != 'wild_test':
                        labels_batch.append(data_frame['label_id'].values[index])
                samples_batch = np.array(samples_batch)
                if mode != 'test' and mode != 'wild_test':
                    labels_batch = to_categorical(labels_batch, num_classes=len(self.label_set))
                    yield (samples_batch, labels_batch)
                else:
                    yield samples_batch
