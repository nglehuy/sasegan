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

import tensorflow as tf

from tensorflow_asr.featurizers.speech_featurizers import read_raw_audio

from .train_dataset import SeganAugTrainDataset, SeganTrainDataset
from ..featurizers.speech_featurizer import SpeechFeaturizer


class SeganAugTestDataset(SeganAugTrainDataset):
    def __init__(self,
                 speech_featurizer: SpeechFeaturizer,
                 clean_dir: str,
                 noises_config: dict):
        super(SeganAugTestDataset, self).__init__(
            stage="test", speech_featurizer=speech_featurizer, clean_dir=clean_dir, noises_config=noises_config)

    def parse(self, clean_wav):
        noisy_wav = self.noises.augment(clean_wav)
        noisy_slices = self.speech_featurizer.extract(noisy_wav)
        clean_slices = self.speech_featurizer.extract(clean_wav)
        return clean_slices, noisy_slices

    def create(self):
        def _gen_data():
            for clean_wav_path in self.data_paths:
                clean_wav = read_raw_audio(clean_wav_path, sample_rate=self.speech_featurizer.sample_rate)
                clean_slices, noisy_slices = self.parse(clean_wav)
                yield clean_wav_path, clean_slices, noisy_slices

        dataset = tf.data.Dataset.from_generator(
            _gen_data,
            output_types=(tf.string, tf.float32),
            output_shapes=(
                tf.TensorShape([]),
                tf.TensorShape([None, *self.speech_featurizer.shape]),
                tf.TensorShape([None, *self.speech_featurizer.shape])
            )
        )
        # Prefetch to improve speed of input length
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


class SeganTestDataset(SeganTrainDataset):
    def __init__(self,
                 speech_featurizer: SpeechFeaturizer,
                 clean_dir: str,
                 noisy_dir: str):
        super(SeganTestDataset, self).__init__(
            stage="test", speech_featurizer=speech_featurizer, clean_dir=clean_dir, noisy_dir=noisy_dir)

    def parse(self, clean_wav, noisy_wav):
        clean_slices = self.speech_featurizer.extract(clean_wav)
        noisy_slices = self.speech_featurizer.extract(noisy_wav)
        return clean_slices, noisy_slices

    def create(self):
        def _gen_data():
            for clean_wav_path in self.data_paths:
                clean_wav = read_raw_audio(clean_wav_path, sample_rate=self.speech_featurizer.sample_rate)
                noisy_wav_path = clean_wav_path.replace(self.clean_dir, self.noisy_dir)
                noisy_wav = read_raw_audio(noisy_wav_path, sample_rate=self.speech_featurizer.sample_rate)
                clean_slices, noisy_slices = self.parse(clean_wav, noisy_wav)
                yield clean_wav_path, clean_slices, noisy_slices

        dataset = tf.data.Dataset.from_generator(
            _gen_data,
            output_types=(tf.string, tf.float32),
            output_shapes=(
                tf.TensorShape([]),
                tf.TensorShape([None, *self.speech_featurizer.shape]),
                tf.TensorShape([None, *self.speech_featurizer.shape])
            )
        )
        # Prefetch to improve speed of input length
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
