import re
import logging

import comfy.utils
import comfy.lora
import comfy.lora_convert
import folder_paths

logger = logging.getLogger("SmartLoRA.Krea2Filter")

_KREA2_BLOCK_RE = re.compile(
    r"diffusion_model[_.]blocks[_.](\d+)[_.](attn|mlp)"
)


def _get_key_weight(key, config):
    m = _KREA2_BLOCK_RE.search(key)
    if m:
        idx        = int(m.group(1))
        layer_type = m.group(2)
        bv         = config["block_weights"]
        block_w    = bv[idx] if idx < len(bv) else 1.0
        layer_w    = config.get(layer_type, 1.0)
        return round(block_w * layer_w, 4)

    # Key doesn't belong to any numbered block (norm layers, etc.)
    return 1.0 if config["other_keys"] else 0.0


class Krea2LoraFilter:
    def __init__(self):
        self.loaded_lora      = None
        self.loaded_lora_path = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":          ("MODEL",),
                "lora_name":      (sorted(folder_paths.get_filename_list("loras")),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "block_weights":  ("KREA2_BLOCK_WEIGHTS",),
                "enabled":        ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "info")
    FUNCTION = "process"
    CATEGORY = "Smart LoRA"

    def _apply(self, model, lora_path, strength, config):
        if lora_path != self.loaded_lora_path or self.loaded_lora is None:
            self.loaded_lora      = comfy.utils.load_torch_file(lora_path, safe_load=True)
            self.loaded_lora_path = lora_path

        key_map = comfy.lora.model_lora_keys_unet(model.model, {})

        try:
            lora_data = comfy.lora_convert.convert_lora(self.loaded_lora)
        except Exception:
            lora_data = self.loaded_lora

        loaded = comfy.lora.load_lora(lora_data, key_map, log_missing=False)

        # Group patches by their computed weight, drop zeros
        groups = {}
        for key, patches in loaded.items():
            w = _get_key_weight(key, config)
            if w != 0.0:
                groups.setdefault(w, {})[key] = patches

        kept    = sum(len(g) for g in groups.values())
        dropped = len(loaded) - kept

        new_model = model.clone()
        for w, patches in groups.items():
            new_model.add_patches(patches, strength * w)

        return new_model, kept, dropped

    def process(self, model, lora_name, strength_model, block_weights, enabled=True):
        if not enabled:
            return (model, "Disabled")

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None:
            return (model, f"LoRA not found: {lora_name}")

        try:
            new_model, kept, dropped = self._apply(model, lora_path, strength_model, block_weights)
            return (new_model, f"{lora_name} | strength:{strength_model:.2f} | {kept} kept, {dropped} filtered")
        except Exception as e:
            logger.error(f"Krea2LoraFilter error: {e}")
            return (model, f"ERROR: {e}")
