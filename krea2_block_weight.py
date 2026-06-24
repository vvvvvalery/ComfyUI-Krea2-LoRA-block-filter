NUM_KREA2_BLOCKS = 28

class Krea2LoraBlockWeight:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "attn":       ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "mlp":        ("FLOAT",   {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            "other_keys": ("BOOLEAN", {"default": True}),
        }
        for i in range(NUM_KREA2_BLOCKS):
            required[f"block_{i:02d}"] = ("BOOLEAN", {"default": True})
        return {"required": required}

    RETURN_TYPES = ("KREA2_BLOCK_WEIGHTS",)
    RETURN_NAMES = ("block_weights",)
    FUNCTION = "configure"
    CATEGORY = "Smart LoRA"

    def configure(self, attn, mlp, other_keys, **kwargs):
        bv = [1.0 if kwargs.get(f"block_{i:02d}", True) else 0.0
              for i in range(NUM_KREA2_BLOCKS)]
        return ({"attn": attn, "mlp": mlp, "other_keys": other_keys, "block_weights": bv},)
