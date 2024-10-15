from typing import Optional

from transformers import AutoModelForCausalLM
from ..base import Actor


class BaichuanActor(Actor):
    """
    Baichuan Actor model.

    Args:
        pretrained (str): Pretrained model name or path.
        config (BaichuanConfig): Model config.
        checkpoint (bool): Enable gradient checkpointing.
        lora_rank (int): LoRA rank.
        lora_train_bias (str): LoRA bias training mode.
    """

    def __init__(
        self,
        pretrained: Optional[str] = None,
        config = None,
        checkpoint: bool = False,
        lora_rank: int = 0,
        lora_train_bias: str = "none",
    ) -> None:
        if pretrained is not None:
            model = AutoModelForCausalLM.from_pretrained(pretrained, trust_remote_code=True)
        elif config is not None:
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        else:
            raise ValueError("Either pretrained or config must be provided.")

        if checkpoint:
            model.gradient_checkpointing_enable()

        super().__init__(model, lora_rank, lora_train_bias)
