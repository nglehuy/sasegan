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

from tiramisu_asr.featurizers.speech_featurizers import deemphasis
from tiramisu_asr.featurizers.speech_featurizers import tf_merge_slices, read_raw_audio
from tiramisu_asr.runners.base_runners import BaseTester
from tiramisu_asr.utils.utils import shape_list


class SeganTester(BaseTester):
    def __init__(self,
                 speech_config: dict,
                 config: dict):
        super(SeganTester, self).__init__(config)
        self.speech_config = speech_config

        self.test_noisy_dir = os.path.join(self.config["outdir"], "test", "noisy")
        self.test_gen_dir = os.path.join(self.config["outdir"], "test", "gen")
        self.test_clean_dir = os.path.join(self.config["outdir"], "test", "clean")

        if not os.path.exists(self.test_noisy_dir): os.makedirs(self.test_noisy_dir)
        if not os.path.exists(self.test_gen_dir): os.makedirs(self.test_gen_dir)
        if not os.path.exists(self.test_clean_dir): os.makedirs(self.test_clean_dir)

    def set_test_data_loader(self, test_dataset):
        """Set train data loader (MUST)."""
        self.clean_dir = test_dataset.clean_dir
        self.test_data_loader = test_dataset.create()

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
        clean_wav_path, noisy_wavs = batch
        g_wavs = self.model([noisy_wavs, self.model.get_z(shape_list(noisy_wavs)[0])],
                            training=False)

        results = tf.numpy_function(
            self._perform, inp=[clean_wav_path, tf_merge_slices(g_wavs),
                                tf_merge_slices(noisy_wavs)],
            Tout=tf.float32
        )

    def _perform(self,
                 clean_wav_path: bytes,
                 gen_signal: np.ndarray,
                 noisy_signal: np.ndarray) -> tf.Tensor:
        clean_wav_path = clean_wav_path.decode("utf-8")
        results = self._compare(clean_wav_path, gen_signal, noisy_signal)
        return tf.convert_to_tensor(results, dtype=tf.float32)

    def _save_to_outdir(self,
                        clean_wav_path: str,
                        gen_signal: np.ndarray,
                        noisy_signal: np.ndarray):
        gen_path = clean_wav_path.replace(self.clean_dir, self.test_gen_dir)
        noisy_path = clean_wav_path.replace(self.clean_dir, self.test_noisy_dir)
        try:
            os.makedirs(os.path.dirname(gen_path))
            os.makedirs(os.path.dirname(noisy_path))
        except Exception:
            pass
        # Avoid differences by writing original wav using sf
        clean_wav = read_raw_audio(clean_wav_path, self.speech_config["sample_rate"])
        sf.write("/tmp/clean.wav", clean_wav, self.speech_config["sample_rate"])
        sf.write(gen_path,
                 gen_signal,
                 self.speech_config["sample_rate"])
        sf.write(noisy_path,
                 noisy_signal,
                 self.speech_config["sample_rate"])
        return gen_path, noisy_path

    def _compare(self,
                 clean_wav_path: str,
                 gen_signal: np.ndarray,
                 noisy_signal: np.ndarray) -> list:
        gen_signal = deemphasis(gen_signal, self.speech_config["preemphasis"])
        noisy_signal = deemphasis(noisy_signal, self.speech_config["preemphasis"])

        gen_path, noisy_path = self._save_to_outdir(clean_wav_path, gen_signal, noisy_signal)

        (pesq_gen, csig_gen, cbak_gen,
         covl_gen, ssnr_gen) = self.composite("/tmp/clean.wav", gen_path)
        (pesq_noisy, csig_noisy, cbak_noisy,
         covl_noisy, ssnr_noisy) = self.composite("/tmp/clean.wav", noisy_path)

        return [pesq_gen, csig_gen, cbak_gen, covl_gen, ssnr_gen,
                pesq_noisy, csig_noisy, cbak_noisy, covl_noisy, ssnr_noisy]

    def finish(self):
        with open(self.test_results, "w", encoding="utf-8") as out:
            for idx, key in enumerate(self.test_metrics.keys()):
                out.write(f"{key} = {self.test_metrics[key].result().numpy():.2f}\n")
