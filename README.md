<h1 align="center">
<p>TiramisuSE :cake:</p>
<p align="center">
<a href="https://github.com/usimarit/TiramisuSE/blob/master/LICENSE">
  <img alt="GitHub" src="https://img.shields.io/github/license/usimarit/TiramisuSE?style=for-the-badge&logo=apache">
</a>
<img alt="python" src="https://img.shields.io/badge/python-%3E%3D3.6-blue?style=for-the-badge&logo=python">
<img alt="tensorflow" src="https://img.shields.io/badge/tensorflow-%3E%3D2.3.0-orange?style=for-the-badge&logo=tensorflow">
<img alt="ubuntu" src="https://img.shields.io/badge/ubuntu-%3E%3D18.04-blueviolet?style=for-the-badge&logo=ubuntu">
</p>
</h1>
<h2 align="center">
<p>The Newest Speech Enhancement in Tensorflow 2</p>
</h2>

<p align="center">
TiramisuSE implements some speech enhancement architectures such as Speech Enhancement Generative Adversarial Network (SEGAN). These models can be converted to TFLite to reduce memory and computation for deployment :smile:
</p>

## What's New?

- Moved from [TiramisuASR](https://github.com/usimarit/TiramisuASR)

## :yum: Supported Models

- **SEGAN** (Refer to [https://github.com/santi-pdp/segan](https://github.com/santi-pdp/segan)), see [examples/segan](./examples/segan)

## Requirements

- Ubuntu distribution (`ctc-decoders` and `semetrics` require some packages from apt)
- Python 3.6+
- Tensorflow 2.2+: `pip install tensorflow`

## Setup Environment and Datasets

Install tensorflow: `pip3 install tensorflow` or `pip3 install tf-nightly` (for using tflite)

Install packages: `python3 setup.py install`

For **setting up datasets**, see [datasets](./tiramisu_se/datasets/README.md)

- For _testing_ **Speech Enhancement Model** (i.e SEGAN), install `octave` and run `./scripts/install_semetrics.sh`

- To enable XLA, run `TF_XLA_FLAGS=--tf_xla_auto_jit=2 $python_train_script`

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

See [examples](./examples/) for some predefined ASR models.

## References & Credits

1. [https://github.com/santi-pdp/segan](https://github.com/santi-pdp/segan)
