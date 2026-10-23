"""Partial transfer of stock YOLO26n-seg weights into the P2 variants."""
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.utils import ROOT as ULTRALYTICS_ROOT

STOCK_YAML = ULTRALYTICS_ROOT / "cfg" / "models" / "26" / "yolo26-seg.yaml"
# Stock layers 0-16 (backbone and top-down neck) keep their meaning in every P2 variant.
# The bottom-up head is rebuilt around the extra P2 level and is never transferred.
STOCK_TRANSFERABLE_LAYERS = 17
# Layers inserted on top of the stock topology; they have no stock counterpart.
INSERTED_MODULES = frozenset({"CBAM"})


def _layers(cfg):
    if not isinstance(cfg, dict):
        cfg = yaml.safe_load(Path(cfg).read_text(encoding="utf-8"))
    return cfg["backbone"] + cfg["head"]


def stock_layer_map(cfg):
    """Map stock YOLO26-seg layer indices to the matching layer indices of a model config.

    ``cfg`` is a model YAML path or an already-loaded config dict. Inserted layers are
    skipped; every mapped pair must agree on repeats, module and arguments.
    """
    stock = _layers(STOCK_YAML)[:STOCK_TRANSFERABLE_LAYERS]
    custom = [(index, layer) for index, layer in enumerate(_layers(cfg)) if layer[2] not in INSERTED_MODULES]
    if len(custom) < len(stock):
        raise ValueError("model config has fewer layers than the transferable stock prefix")
    mapping = {}
    for stock_index, (stock_layer, (custom_index, custom_layer)) in enumerate(zip(stock, custom)):
        if list(stock_layer[1:]) != list(custom_layer[1:]):
            raise ValueError(
                f"stock layer {stock_index} {stock_layer[1:]} does not match model layer {custom_index} {custom_layer[1:]}"
            )
        mapping[stock_index] = custom_index
    return mapping


def _remap_state_key(source_key, layer_map):
    parts = source_key.split(".", 2)
    if len(parts) != 3 or parts[0] != "model" or not parts[1].isdigit():
        return None
    destination_index = layer_map.get(int(parts[1]))
    return None if destination_index is None else f"model.{destination_index}.{parts[2]}"


def build_transfer_report(destination_state, source_state, layer_map):
    """Remap mapped stock layers, then classify tensors by shape compatibility."""
    matched = {}
    mapped = []
    shape_mismatched = []
    skipped_unmapped = []
    considered_destinations = set()
    for source_key, source_tensor in source_state.items():
        destination_key = _remap_state_key(source_key, layer_map)
        if destination_key is None:
            skipped_unmapped.append(source_key)
            continue
        considered_destinations.add(destination_key)
        destination_tensor = destination_state.get(destination_key)
        if destination_tensor is None:
            continue
        if source_tensor.shape == destination_tensor.shape:
            matched[destination_key] = source_tensor
            mapped.append({"source": source_key, "destination": destination_key})
        else:
            shape_mismatched.append({"source": source_key, "destination": destination_key})
    mapped_indices = set(layer_map.values())
    transferable_destination_keys = {
        key for key in destination_state
        if len(key.split(".", 2)) == 3 and key.split(".", 2)[1].isdigit()
        and int(key.split(".", 2)[1]) in mapped_indices
    }
    report = {
        "mapped": sorted(mapped, key=lambda item: item["destination"]),
        "missing_destination": sorted(transferable_destination_keys - considered_destinations),
        "shape_mismatched": sorted(shape_mismatched, key=lambda item: item["destination"]),
        "skipped_unmapped": sorted(skipped_unmapped),
    }
    return report, matched


def load_partial_pretrained(model, weights_path, layer_map):
    """Copy shape-compatible stock tensors into ``model`` (an Ultralytics nn.Module) in place."""
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")
    source_state = YOLO(str(weights_path)).model.float().state_dict()
    report, matched = build_transfer_report(model.state_dict(), source_state, layer_map)
    model.load_state_dict(matched, strict=False)
    print(
        f"[TRANSFER] matched={len(report['mapped'])} missing_destination={len(report['missing_destination'])} "
        f"shape_mismatched={len(report['shape_mismatched'])}"
    )
    return report
