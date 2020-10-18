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

import glob
import os

import tensorflow as tf

from tensorflow_asr.augmentations.augments import SignalNoise
from tensorflow_asr.datasets.base_dataset import BaseDataset
from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio
from tensorflow_asr.utils.utils import preprocess_paths

from ..featurizers.speech_featurizer import SpeechFeaturizer


def merge_dirs(paths: list):
    dirs = []
    for path in paths:
        dirs += glob.glob(os.path.join(path, "**", "*.wav"), recursive=True)
    return dirs


class SeganAugTrainDataset(BaseDataset):
    def __init__(self,
                 stage: str,
                 speech_featurizer: SpeechFeaturizer,
                 clean_dir: str,
                 noises_config: dict,
                 cache: bool = False,
                 shuffle: bool = False):
        self.speech_featurizer = speech_featurizer
        self.clean_dir = preprocess_paths(clean_dir)
        self.noises = SignalNoise() if noises_config is None else SignalNoise(**noises_config)
        super(SeganAugTrainDataset, self).__init__(
            merge_dirs([self.clean_dir]), None, cache, shuffle, stage)

    def parse(self, clean_wav):
        noisy_wav = self.noises.augment(clean_wav)

        clean_slices = self.speech_featurizer.extract(clean_wav)
        noisy_slices = self.speech_featurizer.extract(noisy_wav)

        return clean_slices, noisy_slices

    def create(self, batch_size):
        def _gen_data():
            for clean_wav_path in self.data_paths:
                clean_wav = read_raw_audio(
                    clean_wav_path, sample_rate=self.speech_config["sample_rate"])
                clean_slices, noisy_slices = self.parse(clean_wav)
                for clean, noisy in zip(clean_slices, noisy_slices):
                    yield clean, noisy

        dataset = tf.data.Dataset.from_generator(
            _gen_data,
            output_types=(
                tf.float32,
                tf.float32
            ),
            output_shapes=(
                tf.TensorShape(self.speech_featurizer.shape),
                tf.TensorShape(self.speech_featurizer.shape)
            )
        )

        if self.cache:
            dataset = dataset.cache()

        if self.shuffle:
            dataset = dataset.shuffle(16, reshuffle_each_iteration=True)

        dataset = dataset.batch(batch_size, drop_remainder=True)
        # Prefetch to improve speed of input length
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


class SeganTrainDataset(BaseDataset):
    def __init__(self,
                 stage: str,
                 speech_featurizer: SpeechFeaturizer,
                 clean_dir: str,
                 noisy_dir: str,
                 cache: bool = False,
                 shuffle: bool = False):
        self.speech_featurizer = speech_featurizer
        self.clean_dir = preprocess_paths(clean_dir)
        self.noisy_dir = preprocess_paths(noisy_dir)
        super(SeganTrainDataset, self).__init__(
            merge_dirs([self.clean_dir]), None, cache, shuffle, stage)

    def parse(self, clean_wav, noisy_wav):
        clean_slices = self.speech_featurizer.extract(clean_wav)
        noisy_slices = self.speech_featurizer.extract(noisy_wav)
        return clean_slices, noisy_slices

    def create(self, batch_size):
        def _gen_data():
            for clean_wav_path in self.data_paths:
                noisy_wav_path = clean_wav_path.replace(self.clean_dir, self.noisy_dir)
                clean_wav = read_raw_audio(clean_wav_path,
                                           sample_rate=self.speech_config["sample_rate"])
                noisy_wav = read_raw_audio(noisy_wav_path,
                                           sample_rate=self.speech_config["sample_rate"])
                clean_slices, noisy_slices = self.parse(clean_wav, noisy_wav)
                for clean, noisy in zip(clean_slices, noisy_slices):
                    yield clean, noisy

        dataset = tf.data.Dataset.from_generator(
            _gen_data,
            output_types=(
                tf.float32,
                tf.float32
            ),
            output_shapes=(
                tf.TensorShape(self.speech_featurizer.shape),
                tf.TensorShape(self.speech_featurizer.shape)
            )
        )

        if self.cache:
            dataset = dataset.cache()

        if self.shuffle:
            dataset = dataset.shuffle(16, reshuffle_each_iteration=True)

        dataset = dataset.batch(batch_size, drop_remainder=True)
        # Prefetch to improve speed of input length
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
