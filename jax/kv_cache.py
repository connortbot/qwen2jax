import jax.numpy as jnp

from .config import Qwen2Config

def create_kv_cache(batch_size: int, config: Qwen2Config):
    output = {}
    for i in range(config.n_layers):
        output[f"key_{i}"] = jnp.zeros( (batch_size, SEQUENCE_LENGTH, NUM_HEADS, HEAD_DEPTH))
        output[f"value_{i}"] = jnp.zeros( (batch_size, SEQUENCE_LENGTH, NUM_HEADS, HEAD_DEPTH))
        return output
