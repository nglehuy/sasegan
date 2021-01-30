<h1 align="center">
<p>SASEGAN</p>
<p align="center">
<a href="https://github.com/usimarit/selfattention-segan/blob/master/LICENSE">
  <img alt="GitHub" src="https://img.shields.io/github/license/usimarit/selfattention-segan?style=for-the-badge&logo=apache">
</a>
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.6-blue?style=for-the-badge&logo=python">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3E%3D2.3.0-orange?style=for-the-badge&logo=tensorflow">
<img alt="ubuntu" src="https://img.shields.io/badge/ubuntu-%3E%3D18.04-blueviolet?style=for-the-badge&logo=ubuntu">
</p>
</h1>
<h2 align="center">
<p>Self Attention GAN for Speech Enhancement in Tensorflow 2</p>
</h2>

<p align="center">
This is the TensorFlow 2 Version of Self-Attention Generative Adversarial Network for Speech Enhancement. These models can be converted to TFLite :smile:
</p>

## :yum: Supported Models

- **SEGAN** (Refer to [https://github.com/santi-pdp/segan](https://github.com/santi-pdp/segan)), see [examples/segan](./examples/segan)
- **SASEGAN** (Refer to [https://github.com/pquochuy/sasegan](https://github.com/pquochuy/sasegan)), see [examples/sasegan](./examples/sasegan)

## Setup Environment and Datasets

Install tensorflow: `pip3 install tensorflow` or `pip3 install tf-nightly` (for using tflite)

Install packages: `pip3 install .`

For **setting up datasets**, see [datasets](sasegan/datasets/README.md)

To enable XLA, run `TF_XLA_FLAGS=--tf_xla_auto_jit=2 $python_train_script`

Clean up: `python3 setup.py clean --all` (this will remove `/build` contents)

## Training & Testing

**Example YAML Config Structure**

```yaml
speech_config: ...
model_config: ...
decoder_config: ...
learning_config:
  augmentations: ...
  dataset_config:
    train_paths: ...
    eval_paths: ...
    test_paths: ...
    tfrecords_dir: ...
  optimizer_config: ...
  running_config:
    batch_size: 8
    num_epochs: 20
    outdir: ...
    log_interval_steps: 500
```

See [examples](./examples/) for some running scripts.

## References

```
@article{phan2020sasegan,
  title={Self-Attention Generative Adversarial Network for Speech Enhancement},
  author={H. Phan, Hu. L. Nguyen, O. Y. Chén, P. Koch, N. Q. K. Duong, I. McLoughlin, and A. Mertins},
  journal={arXiv preprint arXiv:2010.09132},
  year={2020}
}
```

1. [Speech Enhancement GAN](https://github.com/santi-pdp/segan)
2. [Improving GANs for Speech Enhancement](https://github.com/pquochuy/idsegan)
3. [Self Attention GAN](https://github.com/brain-research/self-attention-gan)

## Contact

Huy Le Nguyen

Email: nlhuy.cs.16@gmail.com