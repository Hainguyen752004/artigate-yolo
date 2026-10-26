"""Build the baseline networks and load the upstream recipe code from third_party/."""
from functools import lru_cache
import importlib
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
THIRD_PARTY = ROOT / "third_party"
MODELS = ("unet", "malunet", "egeunet")


@lru_cache(maxsize=None)
def _load(path, name):
    path = Path(path)
    if not path.is_file():
        hint = " Run python scripts/fetch_malunet.py first." if name == "malunet" else ""
        raise FileNotFoundError(f"{path} is missing.{hint}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def upstream_utils():
    """EGE-UNet's losses and transforms (identical in MALUNet), see third_party/egeunet/README.md.

    Imported by name (not by path) so data-loader workers started with spawn can unpickle the transforms.
    """
    folder = str(THIRD_PARTY / "egeunet")
    if folder not in sys.path:
        sys.path.append(folder)
    return importlib.import_module("egeunet_utils")


def build_model(name, kwargs):
    if name == "unet":
        from artigate.baselines.unet import UNet

        return UNet(**kwargs)
    if name == "egeunet":
        return _load(THIRD_PARTY / "egeunet" / "egeunet.py", "egeunet").EGEUNet(**kwargs)
    if name == "malunet":
        return _load(THIRD_PARTY / "malunet" / "malunet.py", "malunet").MALUNet(**kwargs)
    raise ValueError(f"unknown baseline {name!r}; expected one of {MODELS}")


def split_output(output):
    """(deep-supervision probabilities or None, main probability) for any of the three networks.

    EGE-UNet with gt_ds returns ((gt_pre5, ..., gt_pre1), out); U-Net and MALUNet return out. All are
    sigmoid probabilities of shape (B, 1, H, W).
    """
    if isinstance(output, tuple):
        return output[0], output[1]
    return None, output
