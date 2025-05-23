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
import tqdm as tqdm

from jax import lax

from .config import Qwen2Config

os.environ['JAX_PLATFORM_NAME'] = 'gpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'true'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    '--xla_gpu_enable_async_all_gather=true'
)
print("JAX devices:", jax.devices())

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

@partial(jax.jit, static_argnames=['dim', 'max_seq_len', 'theta'])
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

@jax.jit
def rotate_half_jax(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate([-x2, x1], axis=-1)

@jax.jit
def apply_rotary_emb(xq, xk, cos, sin, offset=0): # Now takes cos, sin
    """Apply rotary embeddings using cos/sin like PyTorch."""
    # Ensure cos/sin are broadcastable (similar to unsqueeze_dim=1)
    # Assuming xq/xk shape is (B, T, H, D) or similar, need cos/sin like (1, T, 1, D)
    # Adjust slicing/unsqueezing based on actual shapes and how freqs_cis was passed before
    seq_len = xq.shape[1] # Assuming T is the sequence length dimension
    cos = cos[offset:seq_len+offset] # Slice to actual sequence length
    sin = sin[offset:seq_len+offset] # Slice to actual sequence length
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

        # Output projection
        self.wo = nn.Dense(
            dim,
            dtype=self.config.dtype,
            kernel_init=nn.initializers.normal(config.initializer_range),
            use_bias=False
        )

    def __call__(
        self,
        x,
        freqs_cis,
        mask=None,
        deterministic=True,
        layer_idx=-1,
        past_key_values=None,
        use_cache=False,
    ):
        
        B, T, C = x.shape
        n_heads = self.config.n_heads
        n_kv_heads = self.config.n_kv_heads
        head_dim = C // n_heads

        # Linear projections
        q = self.wq(x).reshape(B, T, n_heads, head_dim)
        k = self.wk(x).reshape(B, T, n_kv_heads, head_dim)
        v = self.wv(x).reshape(B, T, n_kv_heads, head_dim)
       
        offset = 0
        if past_key_values is not None:
            # then we need to set an offset, otherwise apply rotary emb
            # will just assume we're at the start.
            offset = past_key_values[0].shape[1]
        q_rope, k_rope = apply_rotary_emb(q, k, freqs_cis[0], freqs_cis[1], offset=offset)
        q = q_rope
        k = k_rope

        if use_cache:
            if past_key_values is not None:
                past_k, past_v = past_key_values
                k = jnp.concatenate([past_k, k], axis=1)
                v = jnp.concatenate([past_v, v], axis=1)
            current_key_value = (k, v)
        else:
            current_key_value = None
        # Repeat k and v heads if n_heads > n_kv_heads (grouped-query attention)
        if n_heads > n_kv_heads:
            k = jnp.repeat(k, n_heads // n_kv_heads, axis=2)
            v = jnp.repeat(v, n_heads // n_kv_heads, axis=2)


        q, k, v = map(lambda x: jnp.swapaxes(x, 1, 2), (q, k, v))
        
        # Adjust attention mask for KV caching if needed
        if use_cache and past_key_values is not None:
            seq_len_k = k.shape[2]  # Length of keys, including cached keys
            
            if mask is not None and mask.shape[-2] < seq_len_k:
                extended_mask = jnp.zeros((B, 1, T, seq_len_k))
                extended_mask = extended_mask.at[:, :, :, -T:].set(mask[:, :, :T, :T])
                mask = extended_mask

        output = flash_attention(q, k, v, mask)

        output = jnp.swapaxes(output, 1, 2).reshape(B, T, -1)
        output = self.wo(output)
        
        if use_cache:
            return output, current_key_value
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

    def __call__(
        self,
        hidden_states,
        freqs_cis,
        attention_mask=None,
        deterministic=True,
        layer_idx=-1,
        past_key_value=None,
        use_cache=False,
    ):

        residual = hidden_states
        norm_input = hidden_states
        normed_hidden_states = self.input_layernorm(norm_input)
        
        if use_cache:
            attn_output, current_key_value = self.self_attn(
                normed_hidden_states,
                freqs_cis,
                attention_mask,
                deterministic,
                layer_idx=layer_idx,
                past_key_values=past_key_value,
                use_cache=use_cache,
            )
        else:
            attn_output = self.self_attn(
                normed_hidden_states,
                freqs_cis,
                attention_mask,
                deterministic,
                layer_idx=layer_idx,
            )
            current_key_value = None

        hidden_states = residual + self.dropout(
            attn_output, # use the tensor output from attention
            deterministic=deterministic
        )

        residual = hidden_states
        mlp_input = self.post_attention_layernorm(hidden_states)
        
        hidden_states = residual + self.dropout(
            self.mlp(mlp_input),
            deterministic=deterministic
        )
        if use_cache:
            return hidden_states, current_key_value
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
        deterministic=True,
        params=None,
        past_key_values=None,
        use_cache=False,
    ):
        B, T = input_ids.shape

        if params is not None and self.apply_weight_tying and not deterministic:
            params = self._tie_weights(params)
        
        if attention_mask is None:
            if use_cache and past_key_values is not None:
                mask = jnp.tril(jnp.ones((T, T)))
                mask = jnp.where(mask == 0, jnp.finfo(jnp.float32).min, 0.0)
                attention_mask = mask[None, None, :, :]
            else:
                mask = jnp.tril(jnp.ones((self.config.max_seq_len, self.config.max_seq_len)))
                mask = jnp.where(mask == 0, jnp.finfo(jnp.float32).min, 0.0)
                attention_mask = mask[None, None, :T, :T]

        # Get embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        current_key_values = [] if use_cache else None

        # Apply transformer blocks
        for i, layer in enumerate(self.layers):
            past_key_value = None if past_key_values is None else past_key_values[i]
            
            if use_cache and current_key_values is not None:
                hidden_states, current_key_value = layer(
                    hidden_states,
                    self.freqs_cis,
                    attention_mask,
                    deterministic,
                    layer_idx=i,
                    past_key_value=past_key_value,
                    use_cache=use_cache
                )
                current_key_values.append(current_key_value)
            else:
                hidden_states = layer(
                hidden_states,
                self.freqs_cis,
                attention_mask,
                deterministic,
                layer_idx=i
            )

        # Apply final normalization
        hidden_states = self.norm(hidden_states)

        # Get logits
        logits = self.lm_head(hidden_states)
        
        if use_cache:
            return logits, current_key_values
        return logits
    
    def _generate_first(self, input_ids, attention_mask=None, deterministic=True):
        return self(
            input_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            use_cache=True,
        )

    def _generate_rest(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        deterministic=True,
    ):
        return self(
            input_ids,
            attention_mask=attention_mask,
            deterministic=deterministic,
            past_key_values=past_key_values,
            use_cache=True,
        )

    def generate(
        self,
        input_ids,
        max_new_tokens: int = 20,
        temperature=None,
        top_k=None,
        top_p=None,
        rng_key=None,
        debug=False,
    ):
        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p

        B, T = input_ids.shape

        if rng_key is None:
            rng_key = random.PRNGKey(0)

        output = input_ids
        logits, past_key_values = self._generate_first(input_ids, deterministic=True)

        rng_keys = random.split(rng_key, max_new_tokens) # pre split more efficient apparently

        if debug:
            iterator = tqdm.tqdm(range(max_new_tokens), desc="Generating tokens")
        else:
            iterator = range(max_new_tokens)

        for i in iterator:
            next_token = _sample_next_token(
                logits, 
                rng_keys[i], 
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p
            )
           
            # Append the sampled token to the sequence
            output = jnp.concatenate([output, next_token[:, None]], axis=1)
            
            # Process only the new token with cached KVs
            next_token_input = next_token[:, None]  # Shape [B, 1]
            
            # Forward pass with KV cache
            logits, past_key_values = self._generate_rest(
                next_token_input, 
                past_key_values=past_key_values,
                deterministic=True
            )

        return output
    
    def generate_streaming(
        self,
        input_ids,
        max_new_tokens: int = 20,
        temperature=None,
        top_k=None,
        top_p=None,
        rng_key=None,
    ):

        temperature = temperature if temperature is not None else self.config.temperature
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p

        if rng_key is None:
            rng_key = random.PRNGKey(0)

        logits, past_key_values = self._generate_first(input_ids, deterministic=True)

        rng_keys = random.split(rng_key, max_new_tokens)
        
        for i in range(max_new_tokens):
            next_token = _sample_next_token(
                logits, 
                rng_keys[i], 
                temperature=temperature, 
                top_k=top_k, 
                top_p=top_p
            )
            
            yield next_token
            
            next_token_input = next_token[:, None]  # Shape [B, 1]
            logits, past_key_values = self._generate_rest(
                next_token_input, 
                past_key_values=past_key_values,
                deterministic=True
            )
        
        return



def generate_with_chat_template(model, model_params, tokenizer, messages, max_new_tokens=100, temperature=0.7, top_k=20, top_p=0.8, rng_key=None, debug=False, streaming=False):
    """Generate text response using the chat template and KV caching"""
    # Format the chat messages
    formatted_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Encode the formatted prompt
    input_tokens = tokenizer.encode(formatted_prompt, padding=True, return_tensors="np")
    input_tokens_jax = jnp.array(input_tokens, dtype=jnp.int32)
    
    print("Input tokens shape:", input_tokens.shape)
    print(formatted_prompt)

    if streaming:
        stream = model.apply(
            model_params,  # Pass model_params first
            input_tokens_jax,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            rng_key=rng_key,
            method=model.generate_streaming,
        )
        
        for token in stream:
            # Every other token is the actual token we want to decode
            # (the other is just for internal state tracking)
            decoded_token = tokenizer.decode([token.item()])
            if decoded_token:  # Only yield non-empty tokens
                yield decoded_token
        return


    # Generate using the model with KV caching
    output_tokens = model.apply(
        model_params,
        input_tokens_jax,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        rng_key=rng_key,
        method=model.generate,
        debug=debug,
    )
    
    # Decode the generated tokens
    generated_text = tokenizer.decode(output_tokens[0].tolist(), skip_special_tokens=True)
    
    # Extract just the model's response (removing the prompt)
    response = generated_text[len(formatted_prompt):]
   
    return response

@partial(jax.jit, static_argnames=['temperature', 'top_k', 'top_p'])
def _sample_next_token(logits, rng_key, temperature=1.0, top_k=None, top_p=None):
    next_token_logits = logits[:, -1, :]
    
    if temperature > 0.0:
        next_token_logits = next_token_logits / temperature
    
    if top_k is not None and top_k > 0:
        k = min(top_k, next_token_logits.shape[-1])
        top_k_logits, top_k_indices = jax.lax.top_k(next_token_logits, k=k)
        
        mask = jnp.full_like(next_token_logits, jnp.finfo(next_token_logits.dtype).min)
        batch_indices = jnp.arange(next_token_logits.shape[0])[:, None]
        next_token_logits = mask.at[batch_indices, top_k_indices].set(top_k_logits)
    
    if top_p is not None and top_p < 1.0:
        sorted_indices = jnp.argsort(next_token_logits, axis=-1)[:, ::-1]
        sorted_logits = jnp.take_along_axis(next_token_logits, sorted_indices, axis=-1)
        
        sorted_probs = jax.nn.softmax(sorted_logits, axis=-1)
        cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
        
        cutoff_mask = cumulative_probs <= top_p
        cutoff_mask = cutoff_mask.at[:, 0].set(True)
        
        sorted_logits = jnp.where(
            cutoff_mask,
            sorted_logits,
            jnp.finfo(sorted_logits.dtype).min
        )
        
        next_token_logits = jnp.take_along_axis(
            sorted_logits,
            jnp.argsort(sorted_indices, axis=-1),
            axis=-1
        )
    
    if temperature == 0.0:
        next_token = jnp.argmax(next_token_logits, axis=-1)
    else:
        next_token = jax.random.categorical(rng_key, next_token_logits, axis=-1)
    
    return next_token
