# TODO probably we need to fix some imports
# import LLMEngine
from vllm.core.hybridscheduler import HybridScheduler

class HYBRIDLLMEngine(LLMEngine):
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        parallel_config: ParallelConfig,
        scheduler_config: SchedulerConfig,
        distributed_init_method: str,
        placement_group: Optional["PlacementGroup"],
        log_stats: bool,
    ) -> None:
        super(LLMEngine, self).__init__(
            model_config,
            cache_config,
            parallel_config,
            scheduler_config,
            distributed_init_method,
            placement_group,
            log_stats)
        self.scheduler = HybridScheduler(scheduler_config, cache_config)
