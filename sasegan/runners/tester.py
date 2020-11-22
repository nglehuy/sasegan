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

import os
from tqdm import tqdm

import numpy as np
import soundfile as sf
import tensorflow as tf

from tensorflow_asr.runners.base_runners import BaseTester
from tensorflow_asr.utils.utils import shape_list

from ..featurizers.speech_featurizer import SpeechFeaturizer


class SeganTester(BaseTester):
    def __init__(self, config: dict, speech_featurizer: SpeechFeaturizer):
        super(SeganTester, self).__init__(config)
        self.speech_featurizer = speech_featurizer

        self.test_noisy_dir = os.path.join(self.config.outdir, "test", "noisy")
        self.test_gen_dir = os.path.join(self.config.outdir, "test", "gen")
        self.test_clean_dir = os.path.join(self.config.outdir, "test", "clean")

        if not os.path.exists(self.test_clean_dir): os.makedirs(self.test_clean_dir)
        if not os.path.exists(self.test_gen_dir): os.makedirs(self.test_gen_dir)
        if not os.path.exists(self.test_noisy_dir): os.makedirs(self.test_noisy_dir)

    def set_test_data_loader(self, test_dataset):
        """Set train data loader (MUST)."""
        self.clean_dir = test_dataset.clean_dir
        self.test_data_loader = test_dataset.create()

    def run(self, test_dataset):
        self.set_test_data_loader(test_dataset)
        self._test_epoch()
        self._finish()

    def _test_epoch(self):
        if self.processed_records > 0:
            self.test_data_loader = self.test_data_loader.skip(self.processed_records)
        progbar = tqdm(initial=self.processed_records, total=None,
                       unit="batch", position=0, desc="[Test]")
        test_iter = iter(self.test_data_loader)
        while True:
            try:
                self._test_function(test_iter)
            except StopIteration:
                break
            except tf.errors.OutOfRangeError:
                break
            progbar.update(1)

        progbar.close()

    @tf.function
    def _test_function(self, iterator):
        batch = next(iterator)
        self._test_step(batch)

    def _test_step(self, batch):
        # Test only available for batch size = 1
        clean_wav_path, clean_slices, noisy_slices = batch
        gen_slices = self.model(
            [noisy_slices, self.model.get_z(shape_list(noisy_slices)[0])],
            training=False
        )

        tf.numpy_function(
            self._save_to_outdir, inp=[clean_wav_path, clean_slices, gen_slices, noisy_slices],
            Tout=tf.float32
        )

    def _save_to_outdir(self,
                        clean_wav_path: str,
                        clean_slices: np.ndarray,
                        gen_slices: np.ndarray,
                        noisy_slices: np.ndarray):
        """Save wav to outdir
        Assume dataset clean path is /a/b/c/d.wav and clean_dir is /a/b
        Then it would save to outdir/c/d_0.wav, outdir/c/d_1.wav, ...
        Args:
            clean_wav_path (str): path to clean wav in dataset
            clean_slices (np.ndarray): shape [None, window_size]
            gen_slices (np.ndarray): shape [None, window_size]
            noisy_slices (np.ndarray): shape [None, window_size]
        """
        filename = os.path.splitext(clean_wav_path.replace(self.clean_dir, ""))[0]
        for i in range(len(clean_slices)):
            clean = self.speech_featurizer.iextract(clean_slices[i])
            gen = self.speech_featurizer.iextract(gen_slices[i])
            noisy = self.speech_featurizer.iextract(noisy_slices[i])
            sf.write(
                os.path.join(self.test_clean_dir, f"{filename}_{i}.wav"),
                clean,
                self.speech_featurizer.sample_rate
            )
            sf.write(
                os.path.join(self.test_gen_dir, f"{filename}_{i}.wav"),
                gen,
                self.speech_featurizer.sample_rate
            )
            sf.write(
                os.path.join(self.test_noisy_dir, f"{filename}_{i}.wav"),
                noisy,
                self.speech_featurizer.sample_rate
            )

    def _finish(self):
        tf.print("Finish testing model, please use script to evaluate results")
