# SEGAN Datasets

A `SeganAugDataset` is constructed from some path to **clean** `.wav` files and a path to **noise** `.wav` files. While training, it will add noises from noisy audio files into clean audio signals _on the fly_ according to your configuration :kissing_smiling_eyes:

A `SeganDataset` is constructed from a directory containing **clean** `.wav` and a directory containing **noisy** `.wav`.

**Inputs**

```python
class SeganAugTrainDataset(BaseDataset):
    def __init__(self,
                 stage: str,
                 clean_dir: str,
                 noisy_dir: str,
                 speech_config: dict,
                 cache: bool = False,
                 shuffle: bool = False)
                 
class SeganTrainDataset(BaseDataset):
    def __init__(self,
                 stage: str,
                 clean_dir: str,
                 noisy_dir: str,
                 speech_config: dict,
                 cache: bool = False,
                 shuffle: bool = False):
```

**Outputs when iterating for training**

```python
(clean_wav_slices, noisy_wav_slices)
```

**Outputs when iterating for testing**

```python
(clean_wav_path, noisy_wavs)
```
