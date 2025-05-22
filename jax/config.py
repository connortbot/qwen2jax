from dataclasses import dataclass
import jax.numpy as jnp

@dataclass(frozen=True)
class Qwen2Config:
    """Configuration for Qwen2 model"""
    vocab_size: int = 151936
    dim: int = 896  # "hidden_size" as typically labeled in im the centre of the universe huggingface
    n_layers: int = 24  # "num_hidden_layers"
    n_heads: int = 14  # "num_attention_heads"
    n_kv_heads: int = 2  # "num_key_value_heads" for gqa
    hidden_dim: int = 4864 # "intermediate_size", in Qwen2.5-0.5b its 5.4x hidden_size, or dim*5.4
    hidden_act: str = "silu" # "hidden_act", activation function, in Qwen2.5-0.5b its silu whereas llama2 is stupid swiglu
    max_seq_len: int = 32768  # "max_position_embeddings", Maximum sequence length, or "max_position_embeddings"

    dropout_rate: float = 0.0  # "attention_dropout"

    initializer_range: float = 0.02 # used for weight init
    dtype: jnp.dtype = jnp.bfloat16  # Using bfloat16 for TPU optimization

    # RoPE settings
    rope_theta: float = 1000000.0  # Base for rotary embeddings

    # Attention settings
    rms_norm_eps: float = 1e-6 # rms norm epsilon
    use_cache: bool = True
    use_sliding_window: bool = False
    sliding_window: int = 32768
    max_window_layers: int = 21

    # Training settings, idk what to set defaults for these to
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 100000

    # Generation settings copied from qwen-0.5b-instruct generation config on HF
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.8

