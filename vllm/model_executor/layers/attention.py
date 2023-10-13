"""Multi-head attention."""

from typing import Any, Dict, List, Optional
import torch
import time
import bz2
import zfpy
from threading import Thread
import os
import numpy as np
import torch.nn as nn
from datetime import datetime
from xformers import ops as xops
from xformers.ops.fmha.attn_bias import (BlockDiagonalCausalMask,
                                         LowerTriangularMaskWithTensorBias)

from vllm import attention_ops
from vllm import cache_ops
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.rotary_embedding import (
    DynamicNTKScalingRotaryEmbedding, LinearScalingRotaryEmbedding,
    RotaryEmbedding)

_SUPPORTED_HEAD_SIZES = [64, 80, 96, 112, 128, 256]


def get_elapsed_ms(t0, t1=None):
    if t1 is None:
        return (datetime.now() - t0).total_seconds() * 1000.0
    return (t1 - t0).total_seconds() * 1000.0


global compress_workers
compress_workers = []

class KVCompressMetrics():
    compress_num = 0
    compress_ms = 0.0
    overhead_num = 0
    overhead_ms = 0.0
    max_workers = 0
    key_kbs = 0.0
    val_kbs = 0.0
    key_compress_kbs = 0.0
    val_compress_kbs = 0.0

    def get_key_compression(self):
        if self.key_kbs == 0:
            return 100.0
        return 100.0 * self.key_compress_kbs / self.key_kbs

    def get_val_compression(self):
        if self.val_kbs == 0:
            return 100.0
        return 100.0 * self.val_compress_kbs / self.val_kbs


global kv_compress_metrics
kv_compress_metrics = KVCompressMetrics()


def wait_for_worker(workers, cur_layer, config):
    if len(workers) == 0:
        return False
    if cur_layer == len(workers) - 1:
        return True
    if len(workers) >= KVCompressThread.MAX_THREADS:
        return True
    if not workers[0].is_alive():
        return True
    return False


class KVCompressThread(Thread):
    MAX_THREADS = 96

    def __init__(self, config, gpu_index, cur_layer, key_tensors, val_tensors):
        Thread.__init__(self)
        self.config = config
        self.gpu_index = gpu_index
        self.cur_layer = cur_layer
        self.key_tensors = key_tensors
        self.val_tensors = val_tensors

    def run(self):
        t0 = datetime.now()

        key_shape = self.key_tensors.shape
        val_shape = self.val_tensors.shape

        if self.config.compress_delta > 0: # bzip2
            file_extension = 'bz2'
            knob_name = 'delta'
            knob_value = self.config.compress_delta
        else: # zfpy
            file_extension = 'zfp'
            knob_name = 'tolerance'
            knob_value = -0.1*self.config.compress_delta

        key_filename = f'./kv_cache/key_{self.gpu_index}_{self.cur_layer:03d}.pt.{file_extension}'
        val_filename = f'./kv_cache/val_{self.gpu_index}_{self.cur_layer:03d}.pt.{file_extension}'

        # KV compress
        try:
            if self.config.compress_delta > 0: # bzip2
                self.output_compressed_tensors_bz2(self.key_tensors, key_filename, compress_delta=knob_value)
                self.output_compressed_tensors_bz2(self.val_tensors, val_filename, compress_delta=knob_value)
            else: # zfpy
                self.output_compressed_tensors_zfpy(self.key_tensors, key_filename, tolerance=knob_value)
                self.output_compressed_tensors_zfpy(self.val_tensors, val_filename, tolerance=knob_value)
        except Exception as ex:
            print("Compressing/decompressing KV cache", ex)

        t1 = datetime.now()

        global kv_compress_metrics
        kv_compress_metrics.compress_num += 1
        kv_compress_metrics.compress_ms += get_elapsed_ms(t0, t1)

        # Some logging
        key_kbs = 2*len(self.key_tensors.flatten()) / 1024.0
        val_kbs = 2*len(self.val_tensors.flatten()) / 1024.0
        key_compress_kbs = os.path.getsize(key_filename) / 1024.0
        val_compress_kbs = os.path.getsize(val_filename) / 1024.0
        kv_compress_metrics.key_kbs += key_kbs
        kv_compress_metrics.val_kbs += val_kbs
        kv_compress_metrics.key_compress_kbs += key_compress_kbs
        kv_compress_metrics.val_compress_kbs += val_compress_kbs

        if False and self.gpu_index == 0:
            print(f"[{datetime.now()}] K {key_filename} {key_kbs:5.1f}KB->{key_compress_kbs:6.1f}KB[{100.0*key_compress_kbs/key_kbs:4.1f}%] {self.key_tensors.shape}")
            print(f"[{datetime.now()}] V {val_filename} {val_kbs:5.1f}KB->{val_compress_kbs:6.1f}KB[{100.0*val_compress_kbs/val_kbs:4.1f}%] {self.val_tensors.shape}")
            print(f"[{datetime.now()}] {file_extension} {knob_name}={knob_value} {get_elapsed_ms(t0):.2f}ms")

    def output_compressed_tensors_bz2(self, tensors, filename, compress_delta=0):
        tensors_output = tensors.numpy()
        tensors_output = tensors_output.flatten()
        tensors_output = tensors_output + compress_delta
        tensors_output = tensors_output.astype(np.float16)
        with bz2.open(filename, 'wb') as output_file:
            np.save(output_file, tensors_output)

    def input_compressed_tensors_bz2(self, filename, tensor_shape, compress_delta=32):
        with bz2.open(filename, "rb") as input_file:
            tensors_input = np.load(input_file)
            tensors_input = tensors_input - compress_delta
            tensors_input = tensors_input.astype(np.float16)
            tensors_input = torch.from_numpy(tensors_input)
            tensors_input = torch.reshape(tensors_input, tensor_shape)
        return tensors_input

    def output_compressed_tensors_zfpy(self, tensors, filename, tolerance=1.0):
        tensors_output = tensors.numpy().astype(np.float32)
        compressed_data = zfpy.compress_numpy(tensors_output, tolerance=tolerance)
        with open(filename, 'wb') as output_file:
            output_file.write(compressed_data)

    def input_compressed_tensors_zfpy(self, filename):
        with open(filename, "rb") as input_file:
            tensors_input = zfpy.decompress_numpy(input_file.read()).astype(np.float16)
            tensors_input = torch.from_numpy(tensors_input)
        return tensors_input

    def output_compressed_tensors(self, tensors, filename, delta=0, tolerance=0):
        """
        Output compressed tensors into a file.
        """
        return self.output_compressed_tensors_zfpy(tensors, filename)

    def input_compressed_tensors(self, filename, tensor_shape, delta=0, tolerance=0):
        """
        Input compressed tensors.
        """
        return self.input_compressed_tensors_zfpy(filename)

    def decompress(self):
        # TODO
        key_tensors_input = self.input_compressed_tensors_bz2(key_filename, key_shape, compress_delta=knob_value)
        val_tensors_input = self.input_compressed_tensors_bz2(val_filename, val_shape, compress_delta=config.compress_delta)

        key_tensors_input = self.input_compressed_tensors_zfpy(key_filename)
        val_tensors_input = self.input_compressed_tensors_zfpy(val_filename)

        # Send KV cache to the GPU
        device = torch.device(f"cuda:{str(gpu_index)}")
        key_to_cache = key_tensors_input.to(device)
        torch.cuda.synchronize()
        value_to_cache = val_tensors_input.to(device)
        torch.cuda.synchronize()

        assert key_tensors.shape == key_tensors_input.shape
        assert val_tensors.shape == val_tensors_input.shape

        key_mae = np.average(np.abs(key_tensors - key_tensors_input))
        val_mae = np.average(np.abs(val_tensors - val_tensors_input))

        print(f"[{datetime.now()}] K {key_filename} {key_kbs:5.1f}KB->{key_compress_kbs:6.1f}KB[{100.0*key_compress_kbs/key_kbs:4.1f}%] {key_tensors.shape} {get_elapsed_ms(t1, t2):.2f}+{get_elapsed_ms(t2, t3):.2f}ms MAE:{key_mae:.3f}")
        print(f"[{datetime.now()}] V {val_filename} {val_kbs:5.1f}KB->{val_compress_kbs:6.1f}KB[{100.0*val_compress_kbs/val_kbs:4.1f}%] {val_tensors.shape} {get_elapsed_ms(t3, t4):.2f}+{get_elapsed_ms(t4, t5):.2f}ms MAE:{val_mae:.3f}")


class PagedAttention(nn.Module):
    # pylint: disable=line-too-long
    """GPT-style multi-head PagedAttention.

    This class takes flattened 1D query, key, and value tensors as input. The
    input 1D tensors can either contain prompt tokens or generation tokens, in
    addition to paddings.

    If the input tensors contain prompt tokens, the layout is as follows:

    |<---------------------- num_valid_tokens ---------------------->|
    |<--------------- num_prompt_tokens -------------->|
    |<--prompt_0-->|<--prompt_1-->|...|<--prompt_N-1-->|<--padding-->|

    Otherwise, the layout is as follows:

    |<------------------ num_valid_tokens ------------------->|
    |<------- num_generation_tokens (M) ------->|
    |<--generation_0-->|...|<--generation_M-1-->|<--padding-->|

    The prompts might have different lengths, while the generation tokens always
    have length 1. The paddings are appended to make the input length a multiple
    of 8, which is desirable for Tensor Cores.

    The class does the following:
    1. Perform multi_query_kv_attention for the prompts. This operation does
        not use the KV cache.
    2. Wait for the cache operations (e.g., swap, copy) to finish. The cache
        operations are issued by the cache engine before executing the forward
        pass of the model, and they are executed asynchronously.
    3. Reshape and store the input key and value tensors in the KV cache.
    4. Perform single_query_cached_kv_attention for the generation tokens.
        This operation reads the previous key and value tensors from the KV
        cache.
    5. Output a flattened 1D tensor.
    """

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 scale: float,
                 num_kv_heads: Optional[int] = None,
                 sliding_window: Optional[int] = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.head_mapping = torch.repeat_interleave(
            torch.arange(self.num_kv_heads, dtype=torch.int32, device="cuda"),
            self.num_queries_per_kv)

        if self.head_size not in _SUPPORTED_HEAD_SIZES:
            raise ValueError(f"head_size ({self.head_size}) is not supported. "
                             f"Supported head sizes: {_SUPPORTED_HEAD_SIZES}.")

    def set_attn_bias(
        self,
        input_metadata: InputMetadata,
        dtype: torch.dtype,
    ) -> None:
        del dtype  # Unused.
        if input_metadata.attn_bias:
            # Already set by a previous layer.
            return
        prompt_lens = input_metadata.prompt_lens
        attn_bias = BlockDiagonalCausalMask.from_seqlens(prompt_lens)
        if self.sliding_window is not None:
            attn_bias = attn_bias.make_local_attention(self.sliding_window)
        input_metadata.attn_bias.append(attn_bias)

    def multi_query_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """Normal attention for the prompt tokens.

        Args:
            output: shape = [num_prompt_tokens, num_heads, head_size]
            query: shape = [num_prompt_tokens, num_heads, head_size]
            key: shape = [num_prompt_tokens, num_kv_heads, head_size]
            value: shape = [num_prompt_tokens, num_kv_heads, head_size]
            input_metadata: metadata for paged attention.
        """

        if self.num_kv_heads != self.num_heads:
            # Project the key and value tensors to the desired number of heads.
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=1)
            value = torch.repeat_interleave(value,
                                            self.num_queries_per_kv,
                                            dim=1)

        # TODO(woosuk): The unsqueeze op may incur some CPU overhead. Optimize.
        out = xops.memory_efficient_attention_forward(
            query.unsqueeze(0),
            key.unsqueeze(0),
            value.unsqueeze(0),
            attn_bias=input_metadata.attn_bias[0],
            p=0.0,
            scale=self.scale,
        )
        # TODO(woosuk): Unnecessary copy. Optimize.
        output.copy_(out.squeeze(0))
        return output

    def single_query_cached_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> None:
        """PagedAttention for the generation tokens.

        Args:
            output: shape = [num_generation_tokens, num_heads, head_size]
            query: shape = [num_generation_tokens, num_heads, head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for paged attention.
        """
        block_size = value_cache.shape[3]
        attention_ops.single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            self.head_mapping,
            self.scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
            None,  # alibi_slopes
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: Optional[torch.Tensor],
        value_cache: Optional[torch.Tensor],
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
        config = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """PagedAttention forward pass.

        NOTE: The query, key, and value tensors must be sliced from a qkv
        tensor of shape [num_tokens, 3 * num_heads * head_size].

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for paged attention.
            cache_event: event to wait for the cache operations to finish.

        Returns:
            shape = [num_tokens, num_heads * head_size]
        """

        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        # Pre-allocate the output tensor.
        output = torch.empty_like(query)

        # Compute the attention op for prompts.
        num_prompt_tokens = input_metadata.num_prompt_tokens
        if num_prompt_tokens > 0:
            # Prompt run.
            # if int(torch.cuda.current_device()) == 0:
            #     print("layers prompt run: ", self.layers, num_prompt_tokens)
            #ESHA assert input_metadata.num_generation_tokens == 0
            self.set_attn_bias(input_metadata, dtype=query.dtype)
            self.multi_query_kv_attention(
                output[:num_prompt_tokens],
                query[:num_prompt_tokens],
                key[:num_prompt_tokens],
                value[:num_prompt_tokens],
                input_metadata,
            )

        # Wait until the cache op is done.
        if cache_event is not None:
            cache_event.wait()

        # print("checking generation tokens", input_metadata.num_prompt_tokens, input_metadata.num_generation_tokens)
        # print("Input metadata:", input_metadata.prompt_lens)
        # print("Checking valid tokens", input_metadata.num_valid_tokens)

        # Reshape the keys and values and store them in the cache.
        # When key_cache and value_cache are not provided, the new key
        # and value vectors will not be cached.
        num_valid_tokens = input_metadata.num_valid_tokens
        if (num_valid_tokens > 0 and key_cache is not None
                and value_cache is not None):
            # The stride is 3 because the key and value are sliced from qkv.
            key_to_cache = key[:num_valid_tokens]
            value_to_cache = value[:num_valid_tokens]
            slot_mapping = input_metadata.slot_mapping

            # Handle KV cache compression
            if config is not None and hasattr(config, "compress_delta") and config.compress_delta != 0:
                gpu_index = int(str(key[:num_valid_tokens].device).strip("cuda:"))
                cur_layer = layer_idx if layer_idx else 0

                global compress_workers
                global kv_compress_metrics

                t0 = datetime.now()

                compress_worker = KVCompressThread(
                    config,
                    gpu_index,
                    cur_layer,
                    key_to_cache.cpu(),
                    value_to_cache.cpu(),
                )
                compress_worker.start()
                compress_workers.append(compress_worker)

                if len(compress_workers) > kv_compress_metrics.max_workers:
                    kv_compress_metrics.max_workers = len(compress_workers)
                while wait_for_worker(compress_workers, cur_layer, config):
                   compress_workers.pop(0).join()

                kv_compress_metrics.overhead_num += 1
                kv_compress_metrics.overhead_ms += get_elapsed_ms(t0)

            if input_metadata.to_cache is not None:
                key_to_cache = key_to_cache[input_metadata.to_cache]
                value_to_cache = value_to_cache[input_metadata.to_cache]
                slot_mapping = slot_mapping[input_metadata.to_cache]

            cache_ops.reshape_and_cache(
                key_to_cache,
                value_to_cache,
                key_cache,
                value_cache,
                slot_mapping,
            )

        if input_metadata.num_generation_tokens > 0:
            # Decoding run.
            # ESHA assert input_metadata.num_prompt_tokens == 0
            assert key_cache is not None and value_cache is not None, (
                "key_cache and value_cache must be provided when "
                "generating tokens.")
            # Compute the attention op for generation tokens.
            self.single_query_cached_kv_attention(
                output[num_prompt_tokens:num_valid_tokens],
                query[num_prompt_tokens:num_valid_tokens], key_cache,
                value_cache, input_metadata)

        # Reshape the output tensor.
        # NOTE(woosuk): The output tensor may include paddings.
        return output.view(-1, self.num_heads * self.head_size)


class PagedAttentionWithRoPE(PagedAttention):
    """PagedAttention with rotary positional embedding."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        rotary_dim: int,
        max_position: int = 8192,
        base: int = 10000,
        num_kv_heads: Optional[int] = None,
        is_neox_style: bool = True,
        rope_scaling: Optional[Dict[str, Any]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        super().__init__(num_heads,
                         head_size,
                         scale,
                         num_kv_heads,
                         sliding_window=sliding_window)
        if rope_scaling is None:
            self.rotary_emb = RotaryEmbedding(head_size, rotary_dim,
                                              max_position, base,
                                              is_neox_style)
        else:
            scaling_type = rope_scaling["type"]
            scaling_factor = rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LinearScalingRotaryEmbedding(
                    head_size, rotary_dim, max_position, base, is_neox_style,
                    scaling_factor)
            elif scaling_type == "dynamic":
                self.rotary_emb = DynamicNTKScalingRotaryEmbedding(
                    head_size, rotary_dim, max_position, base, is_neox_style,
                    scaling_factor)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
        config = None,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """ PagedAttention forward pass with rotary embedding.

        Args:
            positions: shape = [num_tokens]
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for paged attention.
            cache_event: event to wait for the cache operations to finish.

        Returns:
            shape = [num_tokens, num_heads * head_size]
        """

        # Apply rotary embedding to the query and key before passing them
        # to the attention op.
        query, key = self.rotary_emb(positions, query, key)
        return super().forward(
            query,
            key,
            value,
            key_cache,
            value_cache,
            input_metadata,
            cache_event,
            config,
            layer_idx,
        )


class PagedAttentionWithALiBi(PagedAttention):
    """PagedAttention with ALiBi attention bias."""

    def __init__(self,
                 num_heads: int,
                 head_size: int,
                 scale: float,
                 slopes: List[float],
                 num_kv_heads: Optional[int] = None) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads)
        assert len(slopes) == num_heads

        slopes = torch.tensor(slopes, dtype=torch.float32)
        self.register_buffer("alibi_slopes", slopes, persistent=False)

    def set_attn_bias(self, input_metadata: InputMetadata,
                      dtype: torch.dtype) -> None:
        if input_metadata.attn_bias:
            # Already set by a previous layer.
            return
        # Generates ALiBi mask for each prompt.
        for prompt_len in input_metadata.prompt_lens:
            bias = torch.arange(prompt_len, dtype=dtype)
            # Note(zhuohan): HF uses
            #     `bias = bias[None, :].repeat(prompt_len, 1)`
            # here. We find that both biases give the same results, but
            # the bias below more accurately follows the original ALiBi
            # paper.
            bias = bias[None, :] - bias[:, None]
            bias = bias.to(self.alibi_slopes.device)

            # When using custom attention bias, xformers requires the bias to
            # be sliced from a tensor whose length is a multiple of 8.
            padded_len = (prompt_len + 7) // 8 * 8
            bias = torch.empty(
                1,  # batch_size
                self.num_heads,
                prompt_len,
                padded_len,
                device=self.alibi_slopes.device,
                dtype=dtype,
            )[:, :, :, :prompt_len].copy_(bias)
            bias.mul_(self.alibi_slopes[:, None, None])
            attn_bias = LowerTriangularMaskWithTensorBias(bias)
            input_metadata.attn_bias.append(attn_bias)

    def multi_query_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        """Attention with ALiBi bias for the prompt tokens.

        Args:
            output: shape = [num_prompt_tokens, num_heads, head_size]
            query: shape = [num_prompt_tokens, num_heads, head_size]
            key: shape = [num_prompt_tokens, num_kv_heads, head_size]
            value: shape = [num_prompt_tokens, num_kv_heads, head_size]
            input_metadata: metadata for paged attention.
        """
        if self.num_kv_heads != self.num_heads:
            # Project the key and value tensors to the desired number of heads.
            key = torch.repeat_interleave(key, self.num_queries_per_kv, dim=1)
            value = torch.repeat_interleave(value,
                                            self.num_queries_per_kv,
                                            dim=1)

        # FIXME(woosuk): Because xformers does not support dynamic sequence
        # lengths with custom attention bias, we process each prompt one by
        # one. This is inefficient, especially when we have many short prompts.
        start = 0
        for i, prompt_len in enumerate(input_metadata.prompt_lens):
            end = start + prompt_len
            out = xops.memory_efficient_attention_forward(
                query[None, start:end],
                key[None, start:end],
                value[None, start:end],
                attn_bias=input_metadata.attn_bias[i],
                p=0.0,
                scale=self.scale,
            )
            # TODO(woosuk): Unnecessary copy. Optimize.
            output[start:end].copy_(out.squeeze(0))
            start += prompt_len
        return output

    def single_query_cached_kv_attention(
        self,
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        input_metadata: InputMetadata,
    ) -> None:
        """PagedAttention with ALiBi bias for the generation tokens.

        Args:
            output: shape = [num_generation_tokens, num_heads, head_size]
            query: shape = [num_generation_tokens, num_heads, head_size]
            key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
                block_size, x]
            value_cache: shape = [num_blocks, num_kv_heads, head_size,
                block_size]
            input_metadata: metadata for paged attention.
        """
        block_size = value_cache.shape[3]
        attention_ops.single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            self.head_mapping,
            self.scale,
            input_metadata.block_tables,
            input_metadata.context_lens,
            block_size,
            input_metadata.max_context_len,
            self.alibi_slopes,
        )
