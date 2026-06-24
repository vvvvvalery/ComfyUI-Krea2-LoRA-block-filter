from .krea2_block_weight import Krea2LoraBlockWeight
from .krea2_lora_filter import Krea2LoraFilter

NODE_CLASS_MAPPINGS = {
    "Krea2LoraBlockWeight": Krea2LoraBlockWeight,
    "Krea2LoraFilter":      Krea2LoraFilter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2LoraBlockWeight": "Krea 2 LoRA Block Weight",
    "Krea2LoraFilter":      "Krea 2 LoRA Filter (Blocks)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
