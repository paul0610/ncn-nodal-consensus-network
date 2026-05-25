"""Teacher model — generates Q&A pairs for knowledge bootstrap.

Uses a large model (local or cloud) to produce question-answer pairs
about a given topic.  The answers are then verified by the local
consensus pipeline before being written to the graph.
"""

from __future__ import annotations

import json
import re

from loguru import logger
from pydantic import BaseModel

from core.config_loader import Config
from core.models import ModelResponse
from providers.factory import ProviderFactory


class QAPair(BaseModel):
    question: str
    answer: str


PROVIDER_TERMS_URLS: dict[str, str] = {
    "deepseek": "https://platform.deepseek.com/terms",
    "qwen": "https://help.aliyun.com/zh/model-studio/terms",
    "together": "https://www.together.ai/terms-of-service",
    "ollama": "(local — no terms required)",
}

_QA_PROMPT = (
    "Genera {count} preguntas factuales con sus respuestas sobre el tema: {topic}\n\n"
    "Responde ÚNICAMENTE con JSON válido en este formato:\n"
    '{{\n'
    '  "qa_pairs": [\n'
    '    {{"question": "...", "answer": "..."}}\n'
    '  ]\n'
    '}}'
)


class TeacherModel:
    """Wraps a large model provider for Q&A generation."""

    def __init__(self, config: Config) -> None:
        self.config = config
        tc = config.knowledge_bootstrap.teacher
        factory = ProviderFactory(config)
        self._provider = factory.create(tc.provider)
        self._model = tc.model

    async def generate_qa(
        self, topic: str, count: int = 20
    ) -> list[QAPair]:
        """Generate *count* Q&A pairs about *topic*."""
        prompt = _QA_PROMPT.format(count=count, topic=topic)
        response: ModelResponse = await self._provider.complete(
            prompt=prompt,
            model=self._model,
            temperature=0.7,
            timeout=120,
        )
        return _parse_qa(response.content)


# ------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"```(?:json)?\s*([\s\S]*?)```")
_JSON_OBJ = re.compile(r"\{[\s\S]*\}")


def _parse_qa(text: str) -> list[QAPair]:
    try:
        m = _JSON_BLOCK.search(text) or _JSON_OBJ.search(text)
        raw = m.group(1) if m and m.lastindex else (m.group(0) if m else text)
        data = json.loads(raw)
        pairs = data.get("qa_pairs", [])
        return [
            QAPair(question=p["question"], answer=p["answer"])
            for p in pairs
            if p.get("question") and p.get("answer")
        ]
    except Exception as exc:
        logger.warning(f"Failed to parse teacher Q&A: {exc}")
        return []
