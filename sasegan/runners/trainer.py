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
from colorama import Fore

import tensorflow as tf

from ..losses.segan_losses import generator_loss, discriminator_loss
from tensorflow_asr.runners.base_runners import BaseTrainer
from tensorflow_asr.utils.utils import shape_list


class SeganTrainer(BaseTrainer):
    def __init__(self,
                 training_config: dict,
                 strategy: tf.distribute.Strategy = None):
        self.deactivate_l1 = False
        self.deactivate_noise = False
        super(SeganTrainer, self).__init__(config=training_config, strategy=strategy)

    def set_train_metrics(self):
        self.train_metrics = {
            "g_l1_loss": tf.keras.metrics.Mean("train_g_l1_loss", dtype=tf.float32),
            "g_adv_loss": tf.keras.metrics.Mean("train_g_adv_loss", dtype=tf.float32),
            "d_adv_loss": tf.keras.metrics.Mean("train_d_adv_loss", dtype=tf.float32)
        }

    def set_eval_metrics(self):
        self.eval_metrics = {
            "g_l1_loss": tf.keras.metrics.Mean("eval_g_l1_loss", dtype=tf.float32),
            "g_adv_loss": tf.keras.metrics.Mean("eval_g_adv_loss", dtype=tf.float32),
            "d_adv_loss": tf.keras.metrics.Mean("eval_d_adv_loss", dtype=tf.float32)
        }

    def save_model_weights(self):
        self.generator.save_weights(os.path.join(self.config.outdir, "latest.h5"))

    def run(self):
        """Run training."""
        if self.steps.numpy() > 0: tf.print("Resume training ...")

        self.train_progbar = tqdm(
            initial=self.steps.numpy(), unit="batch", total=self.total_train_steps,
            position=0, leave=True,
            bar_format="{desc} |%s{bar:20}%s{r_bar}" % (Fore.GREEN, Fore.RESET),
            desc="[Train]"
        )

        while not self._finished():
            self._train_epoch()
            if self.epochs >= self.config.additional_properties["l1_remove_epoch"] \
                    and self.deactivate_l1 is False:
                self.config.additional_properties["l1_lambda"] = 0.
                self.deactivate_l1 = True
            if self.epochs >= self.config.additional_properties["denoise_epoch"] \
                    and self.deactivate_noise is False:
                self.config.additional_properties["noise_std"] *= self.config.additional_properties["noise_decay"]
                if self.config.additional_properties["noise_std"] < self.config.additional_properties["denoise_lbound"]:
                    self.config.additional_properties["noise_std"] = 0.
                    self.deactivate_noise = True

        self.save_checkpoint()

        self.train_progbar.close()
        print("> Finish training")

    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, batch):
        clean_wavs, noisy_wavs = batch
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            z = self.generator.get_z(shape_list(clean_wavs)[0])
            g_clean_wavs = self.generator([noisy_wavs, z], training=True)

            d_real_logit = self.discriminator(
                [clean_wavs, noisy_wavs],
                training=True,
                noise_std=self.config.additional_properties["noise_std"]
            )
            d_fake_logit = self.discriminator(
                [g_clean_wavs, noisy_wavs],
                training=True,
                noise_std=self.config.additional_properties["noise_std"]
            )

            gen_tape.watch(g_clean_wavs)
            disc_tape.watch([d_real_logit, d_fake_logit])

            _gen_l1_loss, _gen_adv_loss = generator_loss(y_true=clean_wavs,
                                                         y_pred=g_clean_wavs,
                                                         l1_lambda=self.config.additional_properties["l1_lambda"],
                                                         d_fake_logit=d_fake_logit)

            _disc_loss = discriminator_loss(d_real_logit, d_fake_logit)

            _gen_loss = _gen_l1_loss + _gen_adv_loss

            train_disc_loss = tf.nn.compute_average_loss(
                _disc_loss, global_batch_size=self.global_batch_size)
            train_gen_loss = tf.nn.compute_average_loss(
                _gen_loss, global_batch_size=self.global_batch_size)

            train_gen_loss = self.generator_optimizer.get_scaled_loss(train_gen_loss)
            train_disc_loss = self.discriminator_optimizer.get_scaled_loss(train_disc_loss)

        gen_grad = gen_tape.gradient(train_gen_loss, self.generator.trainable_variables)
        disc_grad = disc_tape.gradient(train_disc_loss, self.discriminator.trainable_variables)
        gen_grad = self.generator_optimizer.get_unscaled_gradients(gen_grad)
        disc_grad = self.discriminator_optimizer.get_unscaled_gradients(disc_grad)

        self.generator_optimizer.apply_gradients(
            zip(gen_grad, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(
            zip(disc_grad, self.discriminator.trainable_variables))

        self.train_metrics["g_l1_loss"].update_state(_gen_l1_loss)
        self.train_metrics["g_adv_loss"].update_state(_gen_adv_loss)
        self.train_metrics["d_adv_loss"].update_state(_disc_loss)

    @tf.function(experimental_relax_shapes=True)
    def _eval_step(self, batch):
        clean_wavs, noisy_wavs = batch

        z = self.generator.get_z(shape_list(clean_wavs)[0])
        g_clean_wavs = self.generator([noisy_wavs, z], training=False)

        d_real_logit = self.discriminator([clean_wavs, noisy_wavs], training=False)
        d_fake_logit = self.discriminator([g_clean_wavs, noisy_wavs], training=False)

        _gen_l1_loss, _gen_adv_loss = generator_loss(y_true=clean_wavs,
                                                     y_pred=g_clean_wavs,
                                                     l1_lambda=self.config.additional_properties["l1_lambda"],
                                                     d_fake_logit=d_fake_logit)

        _disc_loss = discriminator_loss(d_real_logit, d_fake_logit)

        self.eval_metrics["g_l1_loss"].update_state(_gen_l1_loss)
        self.eval_metrics["g_adv_loss"].update_state(_gen_adv_loss)
        self.eval_metrics["d_adv_loss"].update_state(_disc_loss)

    def compile(self,
                generator: tf.keras.Model,
                discriminator: tf.keras.Model,
                optimizer_config: dict,
                max_to_keep: int = 10):
        with self.strategy.scope():
            self.generator = generator
            self.discriminator = discriminator
            gen_opt = tf.keras.optimizers.get(optimizer_config["generator"])
            disc_opt = tf.keras.optimizers.get(optimizer_config["discriminator"])
            self.generator_optimizer = \
                tf.train.experimental.enable_mixed_precision_graph_rewrite(gen_opt)
            self.discriminator_optimizer = \
                tf.train.experimental.enable_mixed_precision_graph_rewrite(disc_opt)
        self.create_checkpoint_manager(
            max_to_keep, generator=self.generator, gen_optimizer=self.generator_optimizer,
            discriminator=self.discriminator, disc_optimizer=self.discriminator_optimizer
        )
