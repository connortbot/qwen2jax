import jax.numpy as jnp

from .config import Qwen2Config

def create_kv_cache(batch_size: int, config: Qwen2Config):
    output = {}
    head_dim = config.dim // config.n_heads
    for i in range(config.n_layers):
        output[f"key_{i}"] = jnp.zeros( (batch_size, config.max_seq_len, config.n_heads, head_dim))
        output[f"value_{i}"] = jnp.zeros( (batch_size, config.max_seq_len, config.n_heads, head_dim))
        return output
