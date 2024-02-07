from typing import List

from vllm.prefix import Prefix


class PrefixPolicy:

    def get_priority(
        self,
        now: float,
        prefix: Prefix,
    ) -> float:
        raise NotImplementedError

    def sort_by_priority(
        self,
        now: float,
        prefix_list: List[Prefix],
    ) -> List[Prefix]:
        return sorted( 
            prefix_list,
            key=lambda prefix: self.get_priority(now, prefix),
            reverse=True,
        )

class FCFS(PrefixPolicy):

    def get_priority(
        self,
        now: float,
        prefix: Prefix,
    ) -> float:
        for gpu_block in prefix.block_table:
            if gpu_block.ref_count > 1:
                return float('-inf')
        return now - prefix.arrival_time


class LRU(PrefixPolicy):
    
    def get_priority(
        self, 
        now: float, 
        prefix: Prefix 
    ) -> float:
        for gpu_block in prefix.block_table:
            if gpu_block.ref_count > 1:
                return float('-inf')
        return prefix.last_accessed_time


class LFU(PrefixPolicy):
    
    def get_priority(
        self, 
        now: float, 
        prefix: Prefix
    ) -> float:
        for gpu_block in prefix.block_table:
            if gpu_block.ref_count > 1:
                return float('-inf')
        return prefix.freq 

    
class PrefixPolicyFactory:

    _POLICY_REGISTRY = {
        'fcfs': FCFS,
        'lru': LRU,
        'lfu': LFU,
    }

    @classmethod
    def get_policy(cls, policy_name: str, **kwargs) -> PrefixPolicy:
        return cls._POLICY_REGISTRY[policy_name](**kwargs)