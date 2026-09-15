"""Optional neural passage reranker for within-paper search."""

from __future__ import annotations

from dataclasses import dataclass

MS_MARCO_MINILM_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
MS_MARCO_MINILM_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"


@dataclass
class CrossEncoderReranker:
    """Score query/passage pairs with one version-pinned cross encoder."""

    model_id: str = MS_MARCO_MINILM_MODEL
    revision: str = MS_MARCO_MINILM_REVISION
    device: str = "cpu"
    local_files_only: bool = False
    max_length: int = 512
    batch_size: int = 32

    def __post_init__(self):
        from sentence_transformers import CrossEncoder

        # The PyTorch checkpoint also works with older serving environments and avoids a
        # non-contiguous safetensors matmul issue observed with torch 2.10 on Apple Silicon.
        self._model = CrossEncoder(
            self.model_id,
            revision=self.revision,
            device=self.device,
            local_files_only=self.local_files_only,
            max_length=self.max_length,
            model_kwargs={"use_safetensors": False},
        )

    @property
    def config(self) -> dict:
        return {
            "kind": "cross_encoder",
            "model": self.model_id,
            "revision": self.revision,
            "device": self.device,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "weights_format": "pytorch_bin",
        }

    def rank(self, query: str, passages: list[str]) -> list[tuple[int, float]]:
        scores = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return sorted(
            ((index, float(score)) for index, score in enumerate(scores)),
            key=lambda item: (-item[1], item[0]),
        )
