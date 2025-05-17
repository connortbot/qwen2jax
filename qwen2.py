import flax.linen as nn
import jax
from jax import random, lax, vmap
import jax.numpy as jnp
from jax.nn import softmax, silu
from typing import Optional, Tuple
from dataclasses import dataclass
import jax.tree as tree
import os
from functools import partial
import numpy as np

os.environ['JAX_PLATFORM_NAME'] = 'tpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
print("JAX devices:", jax.devices())

# Implementations for many elements, such as RoPE, RMSNorm, etc. taken from:
# https://github.com/dhyaneesh/awesome-jax-flax-llms/tree/main
# most modifications were to adapt to the Qwen2 arch
"""
dim -> hidden_size
hidden_dim -> intermediate_size

Something that confused me when i did solomonoff a while ago and yet again confused me again
The "base dimension for the model" is typically "dim". However, huggingface models call it "hidden_size"

Then, the intermediate_size is the expanded dimension in the multi-layer perceptron.
Llama3 just 4x the original dim, but Qwen2.5-0.5b multiplies it by 5.4x.
THe config below is modified so that the hidden_dim (intermediate size) is specified.
"""

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



#####################
# RMSNorm #
#####################

class RMSNorm(nn.Module):
  """Root Mean Square Layer Normalization"""
  dim: int
  eps: float = 1e-6
  dtype: jnp.dtype = jnp.float32 # bfloat

  @nn.compact
  def __call__(self, x):
    input_dtype = x.dtype
    weight = self.param('weight', nn.initializers.ones, (self.dim,), self.dtype)

    hidden_states = x.astype(jnp.float32)
    variance = jnp.mean(jnp.square(hidden_states), axis=-1, keepdims=True)
    hidden_states = hidden_states * jax.lax.rsqrt(variance + self.eps)
    
    hidden_states = hidden_states * weight

    return hidden_states.astype(input_dtype)


#########################
# Rotary Position Embeddings (RoPE hehe) #
#########################


def precompute_freqs_cis_old(dim: int, max_seq_len: int, theta: float = 1000000.0):
    """Precompute the frequency tensor for complex exponentials (rotary embeddings)."""
    # Compute the frequencies for each feature dimension
    freqs = 1.0 / (theta ** (jnp.arange(0, dim // 2, dtype=jnp.float32) / dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    # Create the frequency matrix by outer product
    freqs = jnp.outer(t, freqs)
    # Convert to complex exponentials
    return jnp.complex64(jnp.exp(1j * freqs))
  
def precompute_freqs_cis(dim: int, max_seq_len: int, theta: float = 1000000.0, config: Qwen2Config = Qwen2Config()):
    """Precompute the frequency tensor for complex exponentials (rotary embeddings)."""
    # Use float32 for intermediate calculations
    inv_freq = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    t = jnp.arange(max_seq_len, dtype=jnp.float32)
    # Create the frequency matrix by outer product
    freqs = jnp.outer(t, inv_freq)
    # Convert to complex exponentials (still okay to use complex64)
    # Concatenate freqs with itself to match the emb = torch.cat((freqs, freqs), dim=-1) structure implicitly
    # before complex exponentiation if using direct complex math.
    # Or calculate cos/sin separately if switching to rotate_half approach.

    # Option A (Keep complex math, ensure freqs covers full dim):
    # freqs_cat = jnp.concatenate([freqs, freqs], axis=-1) # Now shape (max_seq_len, dim)
    # freqs_cis = jnp.exp(1j * freqs_cat).astype(jnp.complex64) # Or just use freqs for cos/sin below

    # Option B (Mimic PyTorch cos/sin approach):
    emb = jnp.concatenate([freqs, freqs], axis=-1) # Shape (max_seq_len, dim)
    cos = jnp.cos(emb).astype(config.dtype) # Cast back to model dtype # bfloat
    sin = jnp.sin(emb).astype(config.dtype) # Cast back to model dtype # bfloat
    return cos, sin # Return cos and sin separately

def rotate_half_jax(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)

def apply_rotary_emb(xq, xk, cos, sin): # Now takes cos, sin
    """Apply rotary embeddings using cos/sin like PyTorch."""
    # Ensure cos/sin are broadcastable (similar to unsqueeze_dim=1)
    # Assuming xq/xk shape is (B, T, H, D) or similar, need cos/sin like (1, T, 1, D)
    # Adjust slicing/unsqueezing based on actual shapes and how freqs_cis was passed before
    seq_len = xq.shape[1] # Assuming T is the sequence length dimension
    cos = cos[:seq_len] # Slice to actual sequence length
    sin = sin[:seq_len] # Slice to actual sequence length
    # Add necessary batch/head dims for broadcasting, e.g.:
    cos = jnp.expand_dims(cos, axis=(0, 2)) # Shape might become (1, T, 1, D)
    sin = jnp.expand_dims(sin, axis=(0, 2)) # Shape might become (1, T, 1, D)

    xq_embed = (xq * cos) + (rotate_half_jax(xq) * sin)
    xk_embed = (xk * cos) + (rotate_half_jax(xk) * sin)
    return xq_embed, xk_embed


#########################
# Flash Attention #
#########################

@partial(jax.jit)
def flash_attention(q, k, v, mask=None, scale=None):
    """
    Optimized implementation of attention mechanism using JAX primitives
    for better compiler optimization and memory efficiency.
    """
    input_dtype = q.dtype
    batch_size, num_heads, seq_len, head_dim = q.shape

    # Compute scale if not provided
    if scale is None:
        scale = 1.0 / jnp.sqrt(head_dim)

    # Compute attention scores with fused operation
    # Fuse transpose and matmul for better compiler optimization
    scores = jnp.einsum('bhid,bhjd->bhij', q, k) * scale

    # Apply causal mask if provided
    if mask is not None:
        scores = scores + mask

    # Stabilize softmax by subtracting max value
    # This prevents overflow and allows for better precision
    scores_f32 = scores.astype(jnp.float32)
    scores_max_f32 = jnp.max(scores_f32, axis=-1, keepdims=True)
    stabilized_scores_f32 = scores_f32 - jax.lax.stop_gradient(scores_max_f32)

    # Apply softmax with higher precision
    attn_weights = jnp.exp(stabilized_scores_f32)
    attn_weights = attn_weights / jnp.sum(attn_weights, axis=-1, keepdims=True)
  
    # back to input type
    attn_weights = attn_weights.astype(input_dtype)
    # Compute attention output using original dtype attn_weights and v
    output = jnp.einsum('bhij,bhjd->bhid', attn_weights, v)

    return output

class QwenCausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with support for grouped-query attention and sliding window attention"""
    config: Qwen2Config

    def setup(self):
        config = self.config
        dim = config.dim
        n_heads = config.n_heads
        n_kv_heads = config.n_kv_heads
        head_dim = dim // n_heads

        # QKV projections
        # changed the dhyaneesh init to match the qwen2 init, using the initializer_range
        self.wq = nn.Dense(
            n_heads * head_dim,
            dtype=self.config.dtype,
            kernel_init=nn.initializers.normal(config.initializer_range),
            use_bias=True
        )
        self.wk = nn.Dense(
            n_kv_heads * head_dim,
            dtype=self.config.dtype,
            kernel_init=nn.initializers.normal(config.initializer_range),
            use_bias=True
        )
        self.wv = nn.Dense(
            n_kv_heads * head_dim,
            dtype=self.config.dtype,
            kernel_init=nn.initializers.normal(config.initializer_range),
            use_bias=True
        )
        # in llama3 its kernel_init=nn.initializers.variance_scaling(1.0, 'fan_in', 'normal')

        # Output projection
        self.wo = nn.Dense(
            dim,
            dtype=self.config.dtype,
            kernel_init=nn.initializers.normal(config.initializer_range),
            use_bias=False
        )

        # Llama3 does QK normalization for stability but i don't think Qwen2 does so
        # self.q_norm = RMSNorm(head_dim)
        # self.k_norm = RMSNorm(head_dim)

    def __call__(
        self,
        x,
        freqs_cis,
        mask=None,
        deterministic=True,
        layer_idx=-1,
        past_key_value=None,
        use_cache=False,
    ):
        
        B, T, C = x.shape
        config = self.config
        n_heads = self.config.n_heads
        n_kv_heads = self.config.n_kv_heads
        head_dim = C // n_heads

        # logging projections
        q_proj = self.wq(x)
        k_proj = self.wk(x)
        v_proj = self.wv(x)

        # Linear projections
        q = self.wq(x).reshape(B, T, n_heads, head_dim)
        k = self.wk(x).reshape(B, T, n_kv_heads, head_dim)
        v = self.wv(x).reshape(B, T, n_kv_heads, head_dim)

        # once again, llama2 does QK normalization for stability but i don't think Qwen2 does so
        # q = jnp.swapaxes(self.q_norm(jnp.swapaxes(q, 1, 2)), 1, 2)
        # k = jnp.swapaxes(self.k_norm(jnp.swapaxes(k, 1, 2)), 1, 2)

        # Apply rotary embeddings
        # q, k = apply_rotary_emb(q, k, freqs_cis[:T])
        q_rope, k_rope = apply_rotary_emb(q, k, freqs_cis[0], freqs_cis[1])
        
        q = q_rope
        k = k_rope

        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = jnp.concatenate([past_key, k], axis=1)
            v = jnp.concatenate([past_value, v], axis=1)
        
        if use_cache:
            present_key_value = (k, v)
        else:
            present_key_value = None

        # Repeat k and v heads if n_heads > n_kv_heads (grouped-query attention)
        if n_heads > n_kv_heads:
            k = jnp.repeat(k, n_heads // n_kv_heads, axis=2)
            v = jnp.repeat(v, n_heads // n_kv_heads, axis=2)

        # Transpose tensors for attention computation (B, H, T, D)
        q, k, v = map(lambda x: jnp.swapaxes(x, 1, 2), (q, k, v))

        # Use flash attention (conceptually)
        output = flash_attention(q, k, v, mask)

        # Transpose output and project back to full dimension
        output = jnp.swapaxes(output, 1, 2).reshape(B, T, -1)
        output = self.wo(output)

        return output


#########################
# Multi-Layer Perceptron #
#########################


class Qwen2MLP(nn.Module):
    """Feed-forward network with Silu activation"""
    config: Qwen2Config
    dtype: jnp.dtype = jnp.float32 # bfloat

    def setup(self):
        config = self.config
        dim = self.config.dim
        hidden_dim = self.config.hidden_dim # llama3 doesn't use the config, it just multiplies dim by 4


        # the mapping in LLaMa3, which has:
        # hidden_dim, dim, hidden_dim (w1, w2, w3))
        # we order it differently to match Qwen2
        # hidden_dim, hidden_dim, dim (w3, w1, w2)
        # functionally equivalent, order of operations is the same
        # Linear projections
        self.gate_proj = nn.Dense(
            hidden_dim,  # from dim -> hidden_dim
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(config.initializer_range),
            use_bias=False
        )
        self.up_proj = nn.Dense(
            hidden_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(config.initializer_range),
            use_bias=False
        )
        self.down_proj = nn.Dense(
            dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(config.initializer_range),
            use_bias=False
        )

    def __call__(self, x):
        return self.down_proj(
            silu(self.gate_proj(x)) * self.up_proj(x)
        )

#########################
# Transformer Block #
#########################


class Qwen2Block(nn.Module):
    """Qwen2 transformer block"""
    config: Qwen2Config

    def setup(self):
        # renamed
        # attention_norm -> input_layernorm
        # ffn_norm -> post_attention_layernorm
        # since attention_norm normalized input before attention (which is input_layernorm)
        # then ffn_norm normalizes AFTER attention, before FFN/MLP
        self.input_layernorm = RMSNorm(
            dim=self.config.dim,
            eps=self.config.rms_norm_eps,
            dtype=self.config.dtype
        )
        self.post_attention_layernorm = RMSNorm(
            dim=self.config.dim,
            eps=self.config.rms_norm_eps,
            dtype=self.config.dtype
        )

        # Attention
        self.self_attn = QwenCausalSelfAttention(self.config)

        # MLP
        self.mlp = Qwen2MLP(self.config, dtype=self.config.dtype)

        # Dropout (but qwen2 uses 0.0 by default)
        self.dropout = nn.Dropout(self.config.dropout_rate)

    def __call__(self, hidden_states, freqs_cis, attention_mask=None, deterministic=True, layer_idx=-1):

        residual = hidden_states
        # To be absolutely clear, let's rename the variable passed to the norm
        norm_input = hidden_states
        normed_hidden_states = self.input_layernorm(norm_input)

        # The input to self_attn is the output of the input_layernorm
        attn_block_input = normed_hidden_states
        
        attn_output_tensor = self.self_attn(
            attn_block_input,
            freqs_cis,
            attention_mask,
            deterministic,
            layer_idx=layer_idx
        )

        hidden_states = residual + self.dropout(
            attn_output_tensor, # use the tensor output from attention
            deterministic=deterministic
        )

        residual = hidden_states
        # To be absolutely clear
        mlp_input = self.post_attention_layernorm(hidden_states)
        
        hidden_states = residual + self.dropout(
            self.mlp(mlp_input),
            deterministic=deterministic
        )

        return hidden_states


#########################
# Transformer: Qwen2 #
#########################

class Qwen2(nn.Module):
    """Qwen2 language model"""
    config: Qwen2Config

    base_model_prefix : str = "model"

    def setup(self):
        config = self.config

        # token embeddings
        self.embed_tokens = nn.Embed(
            config.vocab_size,
            config.dim,
            embedding_init=nn.initializers.normal(stddev=self.config.initializer_range),
            dtype=self.config.dtype
        )

        # Transformer blocks
        self.layers = [
            Qwen2Block(config)
            for _ in range(config.n_layers)
        ]

        # final layer norm
        self.norm= RMSNorm(
            dim=config.dim,
            eps=config.rms_norm_eps,
            dtype=config.dtype
        )

        # Output projection (tied with embeddings)
        self.lm_head = nn.Dense(
            config.vocab_size,
            kernel_init=nn.initializers.normal(stddev=self.config.initializer_range),
            use_bias=False,
            dtype=config.dtype
        )

        # For weight tying
        self.apply_weight_tying = True

        # Pre-compute rotary embeddings
        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len,
            config.rope_theta
        )

    def _tie_weights(self, params):
        """Tie embedding weights with output layer if enabled."""
        if not self.apply_weight_tying:
            return params

        # Create a new parameter dictionary and update lm_head kernel
        new_params = params.copy()
        new_params['lm_head']['kernel'] = new_params['token_embedding']['embedding']
        return new_params

    def __call__(
        self,
        input_ids,
        attention_mask=None, # for sliding window
        position_ids=None,
        deterministic=True,
        params=None
    ):
        # blocks -> layers
        # norm_f -> norm
        # token_embedding -> embed_tokens
        B, T = input_ids.shape

        # Apply weight tying if enabled (only during inference)
        if params is not None and self.apply_weight_tying and not deterministic:
            params = self._tie_weights(params)

        # llama3 creates a max_seq_len x max_seq_len mask
        # but qwen2 creates T x T mask which is the curr seq len (ig more efficient)
        # and qwen2 allows for custom attnetion mask input for sliding window attention
        if attention_mask is None:
            mask = jnp.tril(
            jnp.ones((self.config.max_seq_len, self.config.max_seq_len))
            )
            mask = jnp.where(mask == 0, jnp.finfo(jnp.float32).min, 0.0)
            attention_mask = mask[None, None, :T, :T]

        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)

        # Apply transformer blocks
        for i, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                self.freqs_cis,
                attention_mask,
                deterministic,
                layer_idx=i
            )
            if i == 0:
              layer_0_output_pt = hidden_states

        # Apply final normalization
        hidden_states = self.norm(hidden_states)

        # Get logits
        logits = self.lm_head(hidden_states)

        return logits

    def generate(
        self,
        input_ids,
        max_new_tokens,
        temperature=0.7,
        top_k=20,
        top_p=0.8,
        rng_key=None,
    ):
        temperature = self.config.temperature
        top_k = self.config.top_k
        top_p = self.config.top_p

        """Generate text using the model"""
        B, T = input_ids.shape

        if rng_key is None:
            rng_key = random.PRNGKey(0)

        # Create initial output array
        output = input_ids

        # Generate tokens
        for _ in range(max_new_tokens):
            # Keep the context within max sequence length
            curr_input = output[:, -self.config.max_seq_len:]
            curr_length = curr_input.shape[1]

            # qwen2 reqs making explicit attention mask creation
            mask = jnp.tril(
                jnp.ones((self.config.max_seq_len, self.config.max_seq_len))
            )
            mask = jnp.where(mask == 0, jnp.finfo(jnp.float32).min, 0.0)
            attention_mask = mask[None, None, :T, :T]

            # Get logits for the next token
            logits = self(curr_input, attention_mask=attention_mask, deterministic=True)
            logits = logits[:, -1, :]

            # Apply temperature
            logits = logits / temperature

            # Apply top-k filtering
            if top_k > 0:
                top_k_v, top_k_i = jax.lax.top_k(logits, top_k)
                indices_to_remove = jnp.broadcast_to(
                    jnp.arange(logits.shape[-1]) < top_k_i[:, -1:],
                    logits.shape
                )
                logits = jnp.where(indices_to_remove, logits, jnp.finfo(jnp.float32).min)

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]  # Sort indices in descending order
                sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)  # Get sorted values

                cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep the first token above threshold
                sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1, axis=1)
                sorted_indices_to_remove = sorted_indices_to_remove.at[:, 0].set(False)

                # Scatter sorted tensors to original indexing
                indices_to_remove = jnp.zeros_like(logits, dtype=bool)
                indices_to_remove = indices_to_remove.at[jnp.arange(B)[:, None], sorted_indices].set(sorted_indices_to_remove)

                logits = jnp.where(indices_to_remove, jnp.finfo(self.config.dtype).min, logits)

            # Sample from the filtered distribution
            rng_key, sample_key = random.split(rng_key)
            next_token = random.categorical(sample_key, logits, shape=(B,))

            # Append the sampled token to the sequence
            output = jnp.concatenate([output, next_token[:, None]], axis=1)

        return output
