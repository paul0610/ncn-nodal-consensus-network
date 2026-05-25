"""ReputationSystem — Pareto-weighted model reputation.

The top 20 % of models (by accuracy) produce 80 % of useful claims.
Reputation modifies vote weight in the consensus engine.
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from core.config_loader import Config
from swarm.node import SwarmNode


class ReputationSystem:
    """Track and update per-node reputation scores."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._history_path = Path(config.reputation.history_file)
        self.history: dict[str, list[float]] = self._load_history()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def update(self, model_performances: dict[str, float]) -> None:
        """Update reputation after a session.

        *model_performances* maps ``node_id`` to the ratio of verified
        claims over total claims produced by that node.
        """
        for node_id, performance in model_performances.items():
            if node_id not in self.history:
                self.history[node_id] = []
            self.history[node_id].append(performance)
            self.history[node_id] = self.history[node_id][-100:]

        # Decay inactive nodes
        active = set(model_performances.keys())
        for node_id in set(self.history.keys()) - active:
            decayed = (
                self._current_score(node_id)
                - self.config.reputation.decay_per_session
            )
            self.history[node_id].append(decayed)

        self._save_history()

    def get_score(self, node_id: str) -> float:
        """Return the current reputation score (clamped to [0.1, 1.0])."""
        if node_id not in self.history or not self.history[node_id]:
            return 0.5  # neutral for new nodes
        return max(0.1, min(1.0, self._current_score(node_id)))

    def get_elite_threshold(self, all_nodes: list[SwarmNode]) -> float:
        """Score above which a node is considered *elite* (top 20 %)."""
        if not all_nodes:
            return 0.5
        scores = sorted(
            [self.get_score(n.node_id) for n in all_nodes], reverse=True
        )
        idx = max(0, int(len(scores) * self.config.reputation.elite_ratio) - 1)
        return scores[idx]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _current_score(self, node_id: str) -> float:
        """Exponentially-weighted moving average (recent sessions weigh more)."""
        history = self.history.get(node_id, [])
        if not history:
            return 0.5
        weights = [0.9 ** i for i in range(len(history) - 1, -1, -1)]
        return sum(h * w for h, w in zip(history, weights)) / sum(weights)

    def _load_history(self) -> dict[str, list[float]]:
        if self._history_path.exists():
            try:
                return json.loads(self._history_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning(f"Could not load reputation history: {exc}")
        return {}

    def _save_history(self) -> None:
        try:
            self._history_path.parent.mkdir(parents=True, exist_ok=True)
            self._history_path.write_text(
                json.dumps(self.history, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning(f"Could not save reputation history: {exc}")
