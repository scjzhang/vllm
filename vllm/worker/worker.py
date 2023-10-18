"""A GPU worker class."""
import os
from typing import Dict, List, Tuple, Optional

import torch
import torch.distributed

from mscclpp import Host2HostSemaphore

from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig)
from vllm.model_executor import get_model, InputMetadata, set_random_seed
from vllm.model_executor.parallel_utils.parallel_state import (
    initialize_model_parallel)
from vllm.sampling_params import SamplingParams
from vllm.sequence import SamplerOutput, SequenceData, SequenceGroupMetadata
from vllm.worker.cache_engine import CacheEngine
from vllm.worker.msccl_utils import MscclppGroup, WorkerType
from vllm.utils import get_gpu_memory, get_max_shared_memory_bytes


class Worker:
    """A worker class that executes (a partition of) the model on a GPU.

    Each worker is associated with a single GPU. The worker is responsible for
    maintaining the KV cache and executing the model on the GPU. In case of
    distributed inference, each worker is assigned a partition of the model.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        rank: Optional[int] = None,
        distributed_init_method: Optional[str] = None,
        worker_type: WorkerType = WorkerType.BOTH,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.scheduler_config = scheduler_config
        self.rank = rank
        self.distributed_init_method = distributed_init_method

        # Uninitialized cache engine. Will be initialized by
        # self.init_cache_engine().
        self.cache_config = None
        self.block_size = None
        self.sliding_window = None
        self.cache_engine = None
        self.cache_events = None
        self.gpu_cache = None
        self.msccl_group = None
        self.connections = None
        self.worker_type = worker_type

    def is_token_worker(self):
        return self.worker_type == WorkerType.TOKEN
    
    def is_prompt_worker(self):
        return self.worker_type == WorkerType.PROMPT
    
    def is_both_worker(self):
        return self.worker_type == WorkerType.BOTH

    def init_model(self):
        # This env var set by Ray causes exceptions with graph building.
        os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
        # Env vars will be set by Ray.
        self.rank = self.rank if self.rank is not None else int(
            os.getenv("RANK", "-1"))
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}")
        if self.rank < 0:
            raise ValueError("Invalid or unspecified rank.")
        torch.cuda.set_device(self.device)

        # Initialize the distributed environment.
        _init_distributed_environment(self.parallel_config, self.rank,
                                      self.distributed_init_method)

        # Initialize the model.
        set_random_seed(self.model_config.seed)
        self.model = get_model(self.model_config)

    # call after init_cache in order to have the correct block_size
    def init_msccl_comm(self):
        if not self.parallel_config.sep_prompt_token:
            return

        if self.rank < (self.parallel_config.world_size / 2):
            self.worker_type = WorkerType.PROMPT
        else:
            self.worker_type = WorkerType.TOKEN

        NUM_GPUS = 8
      
        # os.environ['MSCCLPP_DEBUG'] = 'INFO'
        # os.environ['MSCCLPP_DEBUG_SUBSYS'] = 'ALL'
        devices = []
        for gpu_id in range(NUM_GPUS):
            devices.append(f"mlx5_{i}")
        os.environ['MSCCLPP_HCA_DEVICES'] = ','.join(devices)
        os.environ['MSCCLPP_HCA_DEVICES'] = 'mlx5_0,mlx5_1,mlx5_2,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8'
        self.msccl_group = MscclppGroup(
            self.rank,
            self.parallel_config.world_size,
            "eth0:10.0.0.5:51000"
        )
        corr_worker_rank = (self.msccl_group.my_rank + NUM_GPUS) % self.msccl_group.nranks
        self.connections = self.msccl_group.make_connection(
            [corr_worker_rank],
            self.msccl_group.my_ib_device(self.msccl_group.my_rank % NUM_GPUS)
        )

        self.semaphores = {}
        for rank, connection in self.connections.items():
            self.semaphores[rank] = Host2HostSemaphore(self.msccl_group.communicator, connection)

        # register all memory
        self.my_reg_memory = []
        self.token_worker_memory = []
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        mem_size = self.gpu_cache[0][0].numel() * self.gpu_cache[0][0].element_size()
        for layer_id in range(num_layers):
            self.my_reg_memory.append([])
            self.token_worker_memory.append([])
            for k_or_v in [0, 1]:
                reg_mem = self.msccl_group.communicator.register_memory(
                    self.gpu_cache[layer_id][k_or_v].data_ptr(),
                    mem_size,
                    self.msccl_group.my_ib_device(self.msccl_group.my_rank % NUM_GPUS)
                )
                self.my_reg_memory[-1].append(reg_mem)

                tag = layer_id * 2 + k_or_v
                if self.is_prompt_worker():
                    worker_mem = self.msccl_group.communicator.recv_memory_on_setup(
                        corr_worker_rank, tag)
                    self.token_worker_memory[layer_id][k_or_v].append(worker_mem)
                else:
                    self.msccl_group.communicator.send_memory_on_setup(
                        self.my_reg_memory[layer_id][k_or_v], corr_worker_rank, tag)

        self.msccl_group.communicator.setup()

    @torch.inference_mode()
    def profile_num_available_blocks(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        cpu_swap_space: int,
    ) -> Tuple[int, int]:
        # Profile the memory usage of the model and get the maximum number of
        # cache blocks that can be allocated with the remaining free memory.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.

        # Enable top-k sampling to reflect the accurate memory usage.
        vocab_size = self.model.config.vocab_size
        sampling_params = SamplingParams(top_p=0.99, top_k=vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        seqs = []
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            seq_data = SequenceData([0] * seq_len)
            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
            )
            seqs.append(seq)

        input_tokens, input_positions, input_metadata = self._prepare_inputs(
            seqs)

        # Execute the model.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        self.model(
            input_ids=input_tokens,
            positions=input_positions,
            kv_caches=[(None, None)] * num_layers,
            input_metadata=input_metadata,
            cache_events=None,
        )

        # Calculate the number of blocks that can be allocated with the
        # profiled peak memory.
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated()
        total_gpu_memory = get_gpu_memory()
        cache_block_size = CacheEngine.get_cache_block_size(
            block_size, self.model_config, self.parallel_config)
        num_gpu_blocks = int(
            (total_gpu_memory * gpu_memory_utilization - peak_memory) //
            cache_block_size)
        num_cpu_blocks = int(cpu_swap_space // cache_block_size)
        num_gpu_blocks = max(num_gpu_blocks, 0)
        num_cpu_blocks = max(num_cpu_blocks, 0)
        torch.cuda.empty_cache()

        # Reset the seed to ensure that the random state is not affected by
        # the model initialization and profiling.
        set_random_seed(self.model_config.seed)
        return num_gpu_blocks, num_cpu_blocks

    def init_cache_engine(self, cache_config: CacheConfig) -> None:
        self.cache_config = cache_config
        self.block_size = cache_config.block_size
        self.sliding_window = cache_config.sliding_window

        if self.sliding_window is None:
            max_seq_len = self.scheduler_config.max_model_len
        else:
            max_seq_len = min(self.scheduler_config.max_model_len,
                              self.sliding_window)
        _check_if_can_support_max_seq_len(max_seq_len, self.block_size)

        self.cache_engine = CacheEngine(self.cache_config, self.model_config,
                                        self.parallel_config)
        self.cache_events = self.cache_engine.events
        self.gpu_cache = self.cache_engine.gpu_cache

    def _prepare_inputs(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
    ) -> Tuple[torch.Tensor, torch.Tensor, InputMetadata]:
        seq_groups: List[Tuple[List[int], SamplingParams]] = []
        input_tokens: List[int] = []
        input_positions: List[int] = []
        slot_mapping: List[int] = []

        # Add prompt tokens.
        prompt_lens: List[int] = []
        for seq_group_metadata in seq_group_metadata_list:
            if not seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            # Use any sequence in the group.
            seq_id = seq_ids[0]

            seq_data = seq_group_metadata.seq_data[seq_id]
            prompt_tokens = seq_data.get_token_ids()
            prompt_len = len(prompt_tokens)
            prompt_lens.append(prompt_len)

            input_tokens.extend(prompt_tokens)
            # NOTE(woosuk): Here we assume that the first token in the prompt
            # is always the first token in the sequence.
            input_positions.extend(range(len(prompt_tokens)))

            if seq_group_metadata.block_tables is None:
                # During memory profiling, the block tables are not initialized
                # yet. In this case, we just use a dummy slot mapping.
                slot_mapping.extend([0] * prompt_len)
                continue

            # Compute the slot mapping.
            block_table = seq_group_metadata.block_tables[seq_id]
            for i in range(prompt_len):
                block_number = block_table[i // self.block_size]
                block_offset = i % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

        # Add generation tokens.
        max_context_len = 0
        max_num_blocks_per_seq = 0
        context_lens: List[int] = []
        generation_block_tables: List[List[int]] = []
        for seq_group_metadata in seq_group_metadata_list:
            if seq_group_metadata.is_prompt:
                continue

            seq_ids = list(seq_group_metadata.seq_data.keys())
            sampling_params = seq_group_metadata.sampling_params
            seq_groups.append((seq_ids, sampling_params))

            for seq_id in seq_ids:
                seq_data = seq_group_metadata.seq_data[seq_id]
                generation_token = seq_data.get_last_token_id()
                input_tokens.append(generation_token)

                context_len = seq_data.get_len()
                position = context_len - 1
                if self.sliding_window is not None:
                    context_len = min(context_len, self.sliding_window)
                input_positions.append(position)

                block_table = seq_group_metadata.block_tables[seq_id]

                max_context_len = max(max_context_len, context_len)
                max_num_blocks_per_seq = max(max_num_blocks_per_seq,
                                             len(block_table))
                context_lens.append(context_len)

                block_number = block_table[position // self.block_size]
                block_offset = position % self.block_size
                slot = block_number * self.block_size + block_offset
                slot_mapping.append(slot)

                if self.sliding_window is not None:
                    sliding_window_blocks = (self.sliding_window //
                                             self.block_size)
                    block_table = block_table[-sliding_window_blocks:]
                generation_block_tables.append(block_table)

        # Optimization: Pad the input length to be a multiple of 8.
        # This is required for utilizing the Tensor Cores in NVIDIA GPUs.
        input_tokens = _pad_to_alignment(input_tokens, multiple_of=8)
        input_positions = _pad_to_alignment(input_positions, multiple_of=8)

        # Convert to tensors.
        tokens_tensor = torch.tensor(input_tokens,
                                     dtype=torch.long,
                                     device="cuda")
        positions_tensor = torch.tensor(input_positions,
                                        dtype=torch.long,
                                        device="cuda")
        slot_mapping_tensor = torch.tensor(slot_mapping,
                                           dtype=torch.int,
                                           device="cuda")
        context_lens_tensor = torch.tensor(context_lens,
                                           dtype=torch.int,
                                           device="cuda")
        padded_block_tables = [
            _pad_to_max(block_table, max_num_blocks_per_seq)
            for block_table in generation_block_tables
        ]
        block_tables_tensor = torch.tensor(padded_block_tables,
                                           dtype=torch.int,
                                           device="cuda")

        seq_data: Dict[int, SequenceData] = {}
        for seq_group_metadata in seq_group_metadata_list:
            seq_data.update(seq_group_metadata.seq_data)

        input_metadata = InputMetadata(
            seq_groups=seq_groups,
            seq_data=seq_data,
            prompt_lens=prompt_lens,
            slot_mapping=slot_mapping_tensor,
            context_lens=context_lens_tensor,
            max_context_len=max_context_len,
            block_tables=block_tables_tensor,
            sliding_window=self.sliding_window,
        )
        return tokens_tensor, positions_tensor, input_metadata

    @torch.inference_mode()
    def execute_model(
        self,
        seq_group_metadata_list: List[SequenceGroupMetadata],
        blocks_to_swap_in: Dict[int, int],
        blocks_to_swap_out: Dict[int, int],
        blocks_to_copy: Dict[int, List[int]],
        blocks_to_nw: List[int],
        prompt_run: bool = False,
    ) -> SamplerOutput:
        # Issue cache operations.
        issued_cache_op = False
        if blocks_to_swap_in:
            self.cache_engine.swap_in(blocks_to_swap_in)
            issued_cache_op = True
        if blocks_to_swap_out:
            self.cache_engine.swap_out(blocks_to_swap_out)
            issued_cache_op = True
        if blocks_to_copy:
            self.cache_engine.copy(blocks_to_copy)
            issued_cache_op = True

        if issued_cache_op:
            cache_events = self.cache_events
        else:
            cache_events = None

        # If there is no input, we don't need to execute the model.
        if not seq_group_metadata_list:
            if cache_events is not None:
                for event in cache_events:
                    event.wait()
            return {}


        output = {}
        if (prompt_run and self.is_prompt_worker()) or (not prompt_run and self.is_token_worker()) or self.is_both_worker():
            # Prepare input tensors.
            input_tokens, input_positions, input_metadata = self._prepare_inputs(
                seq_group_metadata_list)

            if self.is_prompt_worker():
                prompt_start = time.time()

            # Execute the model.
            output = self.model(
                input_ids=input_tokens,
                positions=input_positions,
                kv_caches=self.gpu_cache,
                input_metadata=input_metadata,
                cache_events=cache_events,
            )

            if self.is_prompt_worker():
                prompt_end = time.time()
                print(f"prompt_time: {prompt_end-prompt_start}", flush=True)
            
        if prompt_run and len(blocks_to_nw):
            self.nw_both(blocks_to_nw, pass_through)
        
        return output

    def nw_both(
        self,
        blocks_to_nw: List[int],
        pass_through: bool,
    ) -> None:
        if pass_through or len(blocks_to_nw) == 0:
            return

        WAIT_TIMEOUT = 10000000000
        COUNT_FLUSH = 1000

        key_block_shape = self.cache_engine.get_key_block_shape()
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        block_num = self.gpu_cache[0][0][0, :].numel()
        block_size = self.gpu_cache[0][0][0, :].element_size()
        mem_size = block_num * block_size
        
        start = time.time()
        count = 0

        # TODO Isn't this just? corr_worker_rank = self.connections.keys[-1]
        for rank in self.connections:
            corr_worker_rank = rank

        for layer_id in range(num_layers):
            # iterate over contiguous kv-blocks
            for block_id in blocks_to_nw:
                # register memory
                for k_or_v in [0, 1]:
                    if self.is_prompt_worker():
                        # write
                        for connection in self.connections.values():
                            connection.write(
                                self.token_worker_memory[layer_id][k_or_v].get(),
                                block_id * mem_size,
                                self.my_reg_memory[layer_id][k_or_v],
                                block_id * mem_size,
                                mem_size
                            )

                        # signal that all writes of this layer are done
                        # TODO would it make sense to signal right after each write?
                        for semaphore in self.semaphores.values():
                            semaphore.signal()
                    else:
                        for sempahore in self.semaphores.values():
                            sempahore.wait(WAIT_TIMEOUT)

                    if count % COUNT_FLUSH == 0:
                        count = 0
                        for connection in self.connections.values():
                            connection.flush()

        for connection in self.connections.values():
            connection.flush()

        end = time.time()
        print(f"time: {end-start}, layers: {num_layers}, k/v: 2, blocks: {len(blocks_to_nw)}, block_size: {key_block_shape}, element_size: {self.gpu_cache[0][0][0, :].element_size()}, blocks: {blocks_to_nw}", flush=True)


def _init_distributed_environment(
    parallel_config: ParallelConfig,
    rank: int,
    distributed_init_method: Optional[str] = None,
) -> None:
    """Initialize the distributed environment."""
    if torch.distributed.is_initialized():
        torch_world_size = torch.distributed.get_world_size()
        if torch_world_size != parallel_config.world_size:
            raise RuntimeError(
                "torch.distributed is already initialized but the torch world "
                "size does not match parallel_config.world_size "
                f"({torch_world_size} vs. {parallel_config.world_size}).")
    elif not distributed_init_method:
        raise ValueError(
            "distributed_init_method must be set if torch.distributed "
            "is not already initialized")
    else:
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=parallel_config.world_size,
            rank=rank,
            init_method=distributed_init_method,
        )

    # A small all_reduce for warmup.
    torch.distributed.all_reduce(torch.zeros(1).cuda())
    initialize_model_parallel(parallel_config.tensor_parallel_size,
                              parallel_config.pipeline_parallel_size,
                              sep_prompt_token=parallel_config.sep_prompt_token)


def _pad_to_alignment(x: List[int], multiple_of: int) -> List[int]:
    return x + [0] * ((-len(x)) % multiple_of)


def _pad_to_max(x: List[int], max_len: int) -> List[int]:
    return x + [0] * (max_len - len(x))


def _check_if_can_support_max_seq_len(max_seq_len: int,
                                      block_size: int) -> None:
    # Follows the logic in
    # attention_kernels.cu::single_query_cached_kv_attention_launcher
    max_shared_mem = get_max_shared_memory_bytes()
    float32_bytes = torch.finfo(torch.float).bits // 8
    padded_max_seq_len = (
        (max_seq_len + block_size - 1) / block_size) * block_size
    # padded_max_seq_len + extra buffer
    required_shared_mem = (padded_max_seq_len + 512) * float32_bytes
    if padded_max_seq_len * float32_bytes > max_shared_mem:
        raise RuntimeError(
            f"vLLM cannot currently support max_model_len={max_seq_len} "
            f"with block_size={block_size} on GPU with compute "
            f"capability {torch.cuda.get_device_capability()} "
            f"(required shared memory {required_shared_mem} > "
            f"available shared memory {max_shared_mem}). "
            "This will be fixed in a future release.")
