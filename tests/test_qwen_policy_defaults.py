"""Keep local Qwen policy decoding settings aligned across comparisons."""

import inspect

from src.policies.grpo_policy import GRPOPolicy
from src.policies.qwen_base_policy import QwenBasePolicy
from src.policies.qwen_common import DEFAULT_MAX_TOKENS, BaseQwenPolicy
from src.policies.qwen_sft_policy import QwenSFTPolicy


def test_all_local_qwen_policies_share_the_generation_budget():
    policies = (BaseQwenPolicy, QwenBasePolicy, QwenSFTPolicy, GRPOPolicy)

    for policy in policies:
        default = inspect.signature(policy.__init__).parameters["max_tokens"].default
        assert default == DEFAULT_MAX_TOKENS == 1024
