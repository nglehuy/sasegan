# Copyright 2020 Huy Le Nguyen (@usimarit) and Huy Phan (@pquochuy)
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
import argparse
from tensorflow_asr.utils import setup_environment, setup_devices

setup_environment()
import tensorflow as tf

DEFAULT_YAML = os.path.join(os.path.abspath(os.path.dirname(__file__)), "config.yml")

parser = argparse.ArgumentParser(prog="SASEGAN")

parser.add_argument("--config", "-c", type=str, default=DEFAULT_YAML,
                    help="The file path of model configuration file")

parser.add_argument("--saved", type=str, default=None,
                    help="Path to saved model")

parser.add_argument("--mxp", default=False, action="store_true",
                    help="Enable mixed precision")

parser.add_argument("--nfx", default=False, action="store_true",
                    help="Choose numpy features extractor")

parser.add_argument("--device", type=int, default=0,
                    help="Device's id to run test on")

args = parser.parse_args()

tf.config.optimizer.set_experimental_options({"auto_mixed_precision": args.mxp})

setup_devices([args.device])

from sasegan.runners.tester import SeganTester
from sasegan.datasets.test_dataset import SeganTestDataset
from tensorflow_asr.configs.config import Config
from sasegan.models.sasegan import Generator
from sasegan.featurizers.speech_featurizer import NumpySpeechFeaturizer, TFSpeechFeaturizer

config = Config(args.config)

speech_featurizer = NumpySpeechFeaturizer(config.speech_config) if args.nfx \
    else TFSpeechFeaturizer(config.speech_config)

tf.random.set_seed(0)
assert args.saved

dataset = SeganTestDataset(
    speech_featurizer=speech_featurizer,
    clean_dir=config.learning_config.dataset_config.clean_test_paths,
    noisy_dir=config.learning_config.dataset_config.noisy_test_paths
)

segan_tester = SeganTester(config.learning_config.running_config)

generator = Generator(window_size=speech_featurizer.window_size, **config.model_config)
generator._build()
generator.load_weights(args.saved, by_name=True)
generator.summary(line_length=100)

segan_tester.compile(generator)
segan_tester.run(dataset)
