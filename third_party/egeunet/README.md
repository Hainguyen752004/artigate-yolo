# EGE-UNet (third-party code)

Source: https://github.com/JCruan519/EGE-UNet at commit `f52ba30c6bf7d0ca479c2c9d4a3cbda999f49d3a`
(2023-07-20). Licence: Apache License 2.0, copied to `LICENSE` in this folder.

| File here | Upstream file | Changes |
|---|---|---|
| `egeunet.py` | `models/egeunet.py` | none (sha256 `c9ae22ca5c6c13a74bb9f0b72958c35708e8b0853541f42e400df6c28ebfc310`) |
| `egeunet_utils.py` | `utils.py` | only `set_seed`, the losses (`BCELoss`, `DiceLoss`, `BceDiceLoss`, `GT_BceDiceLoss`) and the transforms (`myToTensor`, `myResize`, `myRandomHorizontalFlip`, `myRandomVerticalFlip`, `myRandomRotation`, `myNormalize`) are kept, each verbatim; imports reduced |

The training recipe (loss, optimizer, schedule, epochs, batch, augmentation, normalisation) is
transcribed from `configs/config_setting.py`, `train.py` and `engine.py` at the same commit into
`configs/baselines/egeunet.yaml`. Used by `src/artigate/baselines/models.py` and
`scripts/train_seg_baseline.py`. Requires `einops` and `timm`.

Reference: J. Ruan, M. Xie, J. Gao, T. Liu, Y. Fu, "EGE-UNet: an Efficient Group Enhanced UNet for
skin lesion segmentation," MICCAI 2023.
