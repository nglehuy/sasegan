
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

from .train_dataset import SeganAugTrainDataset, SeganTrainDataset
from tiramisu_asr.featurizers.speech_featurizers import preemphasis
from tiramisu_asr.featurizers.speech_featurizers import read_raw_audio, slice_signal


class SeganAugTestDataset(SeganAugTrainDataset):
    def __init__(self,
                 clean_dir: str,
                 noises_config: dict,
                 speech_config: dict):
        super(SeganAugTestDataset, self).__init__(
            "test", clean_dir, noises_config, speech_config)

    def parse(self, clean_wav):
        noisy_wav = self.noises.augment(clean_wav)
        noisy_wav = preemphasis(noisy_wav, self.speech_config["preemphasis"])
        noisy_slices = slice_signal(noisy_wav, self.speech_config["window_size"], 1)
        return noisy_slices

    def create(self):
        def _gen_data():
            for clean_wav_path in self.data_paths:
                clean_wav = read_raw_audio(
                    clean_wav_path, sample_rate=self.speech_config["sample_rate"])
                noisy_slices = self.parse(clean_wav)
                yield (
                    clean_wav_path,
                    noisy_slices
                )

        dataset = tf.data.Dataset.from_generator(
            _gen_data,
            output_types=(tf.string, tf.float32),
            output_shapes=(tf.TensorShape([]),
                           tf.TensorShape([None, self.speech_config["window_size"]]))
        )
        # Prefetch to improve speed of input length
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset


class SeganTestDataset(SeganTrainDataset):
    def __init__(self,
                 clean_dir: str,
                 noisy_dir: str,
                 speech_config: dict):
        super(SeganTestDataset, self).__init__(
            "test", clean_dir, noisy_dir, speech_config)

    def parse(self, noisy_wav):
        noisy_wav = preemphasis(noisy_wav, self.speech_config["preemphasis"])
        noisy_slices = slice_signal(noisy_wav, self.speech_config["window_size"], 1)
        return noisy_slices

    def create(self):
        def _gen_data():
            for clean_wav_path in self.data_paths:
                noisy_wav_path = clean_wav_path.replace(self.clean_dir, self.noisy_dir)
                noisy_wav = read_raw_audio(noisy_wav_path,
                                           sample_rate=self.speech_config["sample_rate"])
                noisy_slices = self.parse(noisy_wav)
                yield (
                    clean_wav_path,
                    noisy_slices
                )

        dataset = tf.data.Dataset.from_generator(
            _gen_data,
            output_types=(tf.string, tf.float32),
            output_shapes=(tf.TensorShape([]),
                           tf.TensorShape([None, self.speech_config["window_size"]]))
        )
        # Prefetch to improve speed of input length
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
