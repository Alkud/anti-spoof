#%%
import os
import numpy as np
import time

from gd_gram_generator import GdGramGenerator
from dataset_generator import DatasetGenerator

import resnet_model

from keras import callbacks

from sklearn.metrics import accuracy_score

from matplotlib.pyplot import plot

from keras.callbacks import TensorBoard


TRAIN_DATA_DIRECTORY = os.path.abspath('')+'/data/train'
TEST_DATA_DIRECTORY = os.path.abspath('')+'/data/test'
LABELS = ['human', 'spoof']
SAMPLE_RATE = 16000
GDGRAM_SHAPE = (512, 256)
NN_INPUT_SHAPE = (GDGRAM_SHAPE[0], GDGRAM_SHAPE[1], 1)
GDGRAM_DURATION = 6.0
INT_DATA_BIT_DEPTH = 16

BATCH = 32
EPOCHS = 150


#%%
# preparing data
image_generator = GdGramGenerator(TRAIN_DATA_DIRECTORY, SAMPLE_RATE, GDGRAM_SHAPE, GDGRAM_DURATION, INT_DATA_BIT_DEPTH)
image_generator.process_input_folder(number_of_threads=50)

#%%
# loading DataFrame with paths/labels for training and validation data and paths for testing data
dataset_generator = DatasetGenerator(label_set=LABELS, train_input_path=TRAIN_DATA_DIRECTORY,
                                     test_input_path=TEST_DATA_DIRECTORY,
                                     bit_depth=INT_DATA_BIT_DEPTH)
data_frame = dataset_generator.load_data()
dataset_generator.apply_train_test_split(test_size=0.3, random_state=911)
dataset_generator.apply_train_validation_split(validation_size=0.2, random_state=74)

#%%
# compiling model
model = resnet_model.build_resnet18(input_shape=NN_INPUT_SHAPE, num_classes=len(LABELS))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['acc'])

#%%
# training model
tensorboard = TensorBoard(log_dir='./logs/{}'.format(int(time.time())), histogram_freq=0,
                          write_graph=True, write_images=False)

callbacks = [callbacks.EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min'), tensorboard]

history = model.fit_generator(generator=dataset_generator.generator(BATCH, mode='train'),
                              steps_per_epoch=int(np.ceil(len(dataset_generator.train_data_frame)/BATCH)),
                              epochs=EPOCHS,
                              verbose=1,
                              callbacks=callbacks,
                              validation_data=dataset_generator.generator(BATCH, mode='validation'),
                              validation_steps=int(np.ceil(len(dataset_generator.validation_data_frame)/BATCH)))

acc = history.history['acc']
plot(acc)

#%%
# testing model
y_pred_proba = model.predict_generator(dataset_generator.generator(BATCH, mode='test'),
                                       int(np.ceil(len(dataset_generator.test_data_frame)/BATCH)),
                                       verbose=1)
y_pred = np.argmax(y_pred_proba, axis=1)

y_true = dataset_generator.test_data_frame['label_id'].values

acc_score = accuracy_score(y_true, y_pred)
print(acc_score)

#%%
# testing in the wild
y_pred_proba = model.predict_generator(dataset_generator.generator(BATCH, mode='wild_test'),
                                       int(np.ceil(len(dataset_generator.wild_test_data_frame)/BATCH)),
                                       verbose=1)

