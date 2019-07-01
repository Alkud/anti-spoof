import os
import math
import random
from glob import glob
import threading
import logging

import numpy as np

from scipy.io import wavfile
from scipy.fftpack import fft
from scipy.signal import hanning

import imageio


class GdGramGenerator:
    def __init__(self, input_path,          # directory containing source .wav files
                 audio_sample_rate,         # all audio files should have the same sample rate!
                 target_image_shape,        # gd-gram image shape
                 gd_gram_duration,          # time duration of gd-gram in seconds
                 target_bit_depth=16,       # number of bits to encode gd-grams
                 build_spectrogram=True,    # if True build and save spectrogram image
                 need_gating=True,          # if True filter from gd-gram points corresponding to low spectrum amplitude
                 gating_threshold=0.01):    # threshold value for gd-gram filtering
        self.input_data_directory = input_path
        self.sample_rate = audio_sample_rate
        assert isinstance(target_image_shape, tuple)  # check whether given image shape is a valid tuple
        self.half_fft_size, self.time_chunks_count = target_image_shape  # fft window size, total number of fft windows
        self.fft_size = 2 * self.half_fft_size
        self.gd_gram_duration = gd_gram_duration
        self.target_data_length = int(self.sample_rate * self.gd_gram_duration)  # corresponding number of audio samples
        self.time_step = math.floor(                                # index increment for overlapping fft windows
            (self.target_data_length - self.fft_size)\
            /(self.time_chunks_count - 1)
        )
        self.coding_bit_depth = target_bit_depth
        self.weighting_window = hanning(self.fft_size)
        self.build_spectrogram = build_spectrogram
        self.need_gating = build_spectrogram and need_gating
        self.gating_threshold = gating_threshold

    def process_audio_files(self, files_to_process, thread_index=0, log_mutex = None):
        processed_file_count = 0
        try:
            for file_path in files_to_process:
                source_dir_name, full_file_name = os.path.split(file_path)
                npy_dir_name = source_dir_name.replace('wav', 'npy')
                file_name, ext = os.path.splitext(full_file_name)
                array_name = '/'.join([npy_dir_name, file_name])
                if os.path.exists(array_name+'.npy'):
                    continue
                gd_gram, spectrogram = self.estimate_gd_gram(file_path)
                np.save(array_name, (gd_gram * (2 ** self.coding_bit_depth - 1)).astype(np.uint16))
                gd_dir_name = source_dir_name.replace('wav', 'tiff/gd')
                image_name = '/'.join([gd_dir_name, file_name + '_gd.tiff'])
                self.save_array_as_tiff(gd_gram, image_name)
                if True == self.build_spectrogram:
                    gd_dir_name = source_dir_name.replace('wav', 'tiff/sp')
                    image_name = '/'.join([gd_dir_name, file_name + '_sp.tiff'])
                    self.save_array_as_tiff(spectrogram, image_name)
                processed_file_count += 1
        except Exception as ex:
            log_mutex.acquire()
            print('Thread# {}: failed:'.format(str(thread_index)), ex.args)
            log_mutex.release()
        finally:
            log_mutex.acquire()
            print('Thread# {0}: terminated, processed {1} files out of {2}'.format(
                str(thread_index), str(processed_file_count), str(len(files_to_process))
            ))
            log_mutex.release()

    def process_input_folder(self, number_of_threads):
        logging.basicConfig(format=format, level=logging.INFO,
                            datefmt="%H:%M:%S")
        print('Processing audio files in', self.input_data_directory)
        file_names = glob(os.path.join(self.input_data_directory + '/wav/', '*/*.wav'))
        print('total files to process:', str(len(file_names)))
        batch_size = math.floor(len(file_names) / number_of_threads)
        thread_pool = list()
        stdout_mutex = threading.Lock()
        for thread_index in range(0, number_of_threads):
            start = thread_index * batch_size
            end = min(start + batch_size, len(file_names))
            current_batch = file_names[start:end]
            current_thread = threading.Thread(
                target=self.process_audio_files,
                args=(current_batch, thread_index, stdout_mutex)
            )
            thread_pool.append(current_thread)
            current_thread.start()
        for thread in thread_pool:
            if thread.is_alive():
                thread.join()

    def estimate_gd_gram(self, file_name) -> (np.ndarray, np.ndarray):
        fs, source_audio = wavfile.read(file_name)
        assert(fs == self.sample_rate)
        source_data_length = len(source_audio)
        # placing source audio randomly in fixed length array of zeros
        audio_data = np.zeros((self.target_data_length,), dtype=np.float)
        source_start_sample = 0
        source_stop_sample = source_data_length
        target_start_sample = 0
        target_stop_sample = self.target_data_length
        if source_data_length < self.target_data_length:
            target_start_sample = random.randint(0, (0.5 * math.floor(self.target_data_length - source_data_length)))
            target_stop_sample = target_start_sample + source_data_length
        elif source_data_length > self.target_data_length:
            source_start_sample = random.randint(0, math.floor(0.5 * (source_data_length - self.target_data_length)))
            source_stop_sample = source_start_sample + self.target_data_length
        assert (target_stop_sample - target_start_sample == source_stop_sample - source_start_sample)
        active_audio_length = target_stop_sample - target_start_sample
        for index in range(0, active_audio_length):
            audio_data[target_start_sample + index] = source_audio[source_start_sample + index]
        # taking overlapping chunks and calculating group delay function
        gd_gram = np.zeros((self.half_fft_size, self.time_chunks_count))
        if True == self.build_spectrogram:
            spectrogram = np.zeros((self.half_fft_size, self.time_chunks_count))
        else:
            spectrogram = np.zeros(1,1)
        slice_indices = self.get_slice_indices()
        chunk_index = 0
        for start in slice_indices:
            stop = start + self.fft_size
            # x(n) - source audio signal
            x = np.zeros(self.fft_size, dtype=np.float)
            # x_(n) = n * x(n) - helper sequence to calculate group delay
            x_ = np.zeros(self.fft_size, dtype=np.float)
            for index in range(start, stop):
                x[index - start] = audio_data[index] * self.weighting_window[index - start]
                x_[index - start] = (index - start) * audio_data[index] * self.weighting_window[index - start]
            # s - complex spectrum of x
            s = fft(x)
            # s_ - complex spectrum of x_
            s_ = fft(x_)
            sr = s.real
            si = s.imag
            sr_ = s_.real
            si_ = s_.imag
            for freq_index in range(0, self.half_fft_size):
                squared_amplitude = sr[freq_index]**2 + si[freq_index]**2
                if squared_amplitude > 0:
                    # gd[i] = (Re(s[i])*Re(s_[i]) + Im(s[i])*Im(s_[i]) / (|s[i]|^2)
                    group_delay = abs(sr[freq_index] * sr_[freq_index] + si[freq_index] * si_[freq_index])\
                                  / squared_amplitude
                    if True == self.build_spectrogram:
                        amplitude = math.sqrt(squared_amplitude)
                else:
                    group_delay = 0
                    if True == self.need_gating:
                        amplitude = 0
                gd_gram[freq_index, chunk_index] = group_delay
                if True == self.build_spectrogram:
                    spectrogram[freq_index, chunk_index] = amplitude
            chunk_index += 1
        if True == self.build_spectrogram:
            max_amplitude_indices = np.unravel_index(np.argmax(spectrogram, axis=None), spectrogram.shape)
            max_amplitude = spectrogram[max_amplitude_indices]
            # normalizing spectrogram
            if max_amplitude > 0:
                for chunk_index in range(0, self.time_chunks_count):
                    for freq_index in range(0, self.half_fft_size):
                        spectrogram[freq_index, chunk_index] /= max_amplitude
        # if gating enabled set all gd-gram values with spectrum amplitudes bellow threshold to zero
        if True == self.need_gating:
            for chunk_index in range(0, self.time_chunks_count):
                for freq_index in range(0, self.half_fft_size):
                    if spectrogram[freq_index, chunk_index] < self.gating_threshold:
                        gd_gram[freq_index, chunk_index] = 0
        max_gd_indices = np.unravel_index(np.argmax(gd_gram, axis=None), gd_gram.shape)
        max_gd = gd_gram[max_gd_indices]
        if max_gd > 0:
            for chunk_index in range(0, self.time_chunks_count):
                for freq_index in range(0, self.half_fft_size):
                    gd_gram[freq_index, chunk_index] /= max_gd
        return gd_gram, spectrogram

    def get_slice_indices(self)->list:
        result = list()
        start = 0
        for i in range(0, self.time_chunks_count):
            result.append(start)
            start = start + self.time_step
        return result

    def save_array_as_data(self, data, path):
        scale_factor = 2 ** self.image_bit_depth

    def save_array_as_tiff(self, data, path):
        scale_factor = 255
        # flip array up-down and convert float range [0..1] to integers [0..scale_factor]
        number_of_rows, number_of_columns = data.shape
        image_array = np.zeros((number_of_rows, number_of_columns), dtype=np.uint8)
        for i in range(0, number_of_rows):
            for j in range(0, number_of_columns):
                image_array[i, j] = int(data[number_of_rows - 1 - i, j] * scale_factor)
        imageio.imwrite(path, image_array)
