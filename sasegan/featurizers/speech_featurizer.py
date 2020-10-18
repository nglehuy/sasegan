# Copyright 2020 Huy Le Nguyen (@usimarit)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import numpy as np
import tensorflow as tf

from tensorflow_asr.featurizers.speech_featurizers import preemphasis, tf_preemphasis
from tensorflow_asr.featurizers.speech_featurizers import depreemphasis, tf_depreemphasis


class SpeechFeaturizer(metaclass=abc.ABCMeta):
    def __init__(self, speech_config: dict):
        """
        speech_config = {
            "sample_rate": int,
            "stride": float,
            "preemphasis": float,
            "window_size": int,
            "pad_end": bool
        }
        """
        # Samples
        self.sample_rate = int(speech_config.get("sample_rate", 16000))
        self.window_size = int(speech_config.get("window_size", 16384))
        self.stride = int(self.window_size * speech_config.get("stride", 1))
        self.preemphasis = float(speech_config.get("preemphasis", 0.95))
        self.pad_end = bool(speech_config.get("pad_end", False))

    @property
    def shape(self) -> list:
        """ The shape of extracted features """
        return [self.window_size]

    @abc.abstractmethod
    def extract(self, signal):
        """ Function to perform feature extraction """
        raise NotImplementedError()

    @abc.abstractmethod
    def iextract(self, slices):
        """ Function to undo feature extraction """
        raise NotImplementedError()


class NumpySpeechFeaturizer(SpeechFeaturizer):

    def extract(self, signal):
        signal = preemphasis(signal, self.preemphasis)
        n_samples = signal.shape[0]
        slices = []
        for beg_i, end_i in zip(range(0, n_samples, self.stride),
                                range(self.window_size, n_samples + self.stride, self.stride)):
            slice_ = signal[beg_i:end_i]
            if slice_.shape[0] < self.window_size:
                if self.pad_end:
                    slice_ = np.pad(slice_, (0, self.window_size - slice_.shape[0]),
                                    'constant', constant_values=0.0)
                else:
                    continue
            if slice_.shape[0] == self.window_size:
                slices.append(slice_)
        return np.array(slices, dtype=np.float32)

    def iextract(self, slices):
        # slices shape = [batch, window_size]
        signal = np.reshape(slices, [-1])
        return depreemphasis(signal, self.preemphasis)


class TFSpeechFeaturizer(SpeechFeaturizer):

    def extract(self, signal):
        signal = tf_preemphasis(signal, self.preemphasis)
        return tf.signal.frame(signal, self.window_size, self.stride,
                               pad_end=self.pad_end, pad_value=0)

    def iextract(self, slices):
        signal = tf.reshape(slices, [-1])
        return tf_depreemphasis(signal, self.preemphasis)
