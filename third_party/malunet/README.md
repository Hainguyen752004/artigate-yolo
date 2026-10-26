# MALUNet (third-party code, not redistributed)

Source: https://github.com/JCruan519/MALUNet at commit `e184b47de99b3fda6b7880fcf3ff99b806308d2b`
(2023-05-24). The repository has no licence, so its code is **not** committed here. Fetch the model
file at that exact commit:

    python scripts/fetch_malunet.py

which downloads `models/malunet.py` to `third_party/malunet/malunet.py` (ignored by git) and checks
its sha256 (`c7c1bacd4f7b434670c97416e490316edeac9b1ab19ec626a070ac6d73308dc0`).

The training recipe is transcribed from `configs/config_setting.py`, `train.py` and `engine.py` at
that commit into `configs/baselines/malunet.yaml`. MALUNet's loss (`BceDiceLoss`) and transforms are
identical to EGE-UNet's, so the Apache-licensed copies in `third_party/egeunet/egeunet_utils.py` are
used. Requires `timm`.

Reference: J. Ruan, S. Xiang, M. Xie, T. Liu, Y. Fu, "MALUNet: A Multi-Attention and Light-weight
UNet for Skin Lesion Segmentation," IEEE BIBM 2022.
