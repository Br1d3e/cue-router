"""Planner-side map of future-position values for intended next shots."""

from dataclasses import dataclass, replace

import numpy as np

from cue_router.core.geometry import HeatmapTarget
from cue_router.core.table import ChineseTableSpecs


@dataclass(frozen=True)
class ContinuationOption:
    target_ball_id: str
    target_pocket_id: str
    route_type: str = "direct"
    score: float = 0.0

    def label(self):
        return f"{self.target_ball_id}:{self.target_pocket_id}:{self.route_type}"


@dataclass
class PositionValueMap:
    value_grid: np.ndarray
    candidates: tuple[ContinuationOption, ...] = ()
    table_width: float = ChineseTableSpecs.TABLE_WIDTH
    table_length: float = ChineseTableSpecs.TABLE_LENGTH
    pocket_mask_radius: float = 1.5 * ChineseTableSpecs.BALL_RADIUS
    best_next_shot: ContinuationOption | None = None
    best_next_shot_index: int | None = None

    def __post_init__(self):
        self.value_grid = np.clip(np.asarray(self.value_grid, dtype=float), 0.0, 1.0)
        if self.value_grid.ndim != 3:
            raise ValueError(f"value_grid must be a 3D array, got shape {self.value_grid.shape!r}")

        self.k, self.m, self.n = self.value_grid.shape
        if len(self.candidates) not in (0, self.k):
            raise ValueError(f"candidates length must be 0 or {self.k}, got {len(self.candidates)}")

        if len(self.candidates) > 0 and all(candidate.score <= 0 for candidate in self.candidates):
            self.candidates = self._score_candidates(self.candidates, self.value_grid, 0.05)

        if self.best_next_shot_index is None:
            self.best_next_shot_index = self._select_best_next_shot_index()
        if self.best_next_shot_index is not None:
            self.best_next_shot = self.candidates[self.best_next_shot_index]

    @classmethod
    def terminal(
            cls,
            m=100,
            n=50,
            pocket_mask_radius=1.5 * ChineseTableSpecs.BALL_RADIUS,
    ):
        heatmap = HeatmapTarget(np.ones((m, n), dtype=float), pocket_mask_radius=pocket_mask_radius)
        cue_X, cue_Y = np.meshgrid(heatmap.x_coords, heatmap.y_coords)
        terminal_values = np.where(heatmap.valid_position_mask(cue_X, cue_Y), 1.0, 0.0)
        return cls(
            terminal_values[np.newaxis, :, :],
            pocket_mask_radius=pocket_mask_radius,
        )

    @classmethod
    def from_evaluator(
            cls,
            evaluator,
            candidates,
            target_ball_pos,
            target_pocket_id,
            obstacles=None,
            m=100,
            n=50,
            pocket_mask_radius=1.5 * ChineseTableSpecs.BALL_RADIUS,
            top_fraction=0.05,
    ):
        heatmap = HeatmapTarget(np.zeros((m, n), dtype=float), pocket_mask_radius=pocket_mask_radius)
        candidates = tuple(candidates)
        value_grid = np.zeros((len(candidates), m, n), dtype=float)

        for row, y in enumerate(heatmap.y_coords):
            for col, x in enumerate(heatmap.x_coords):
                if not heatmap.is_valid_position((x, y)):
                    continue
                value_grid[:, row, col] = evaluator(
                    (x, y),
                    target_ball_pos,
                    target_pocket_id,
                    candidates=candidates,
                    obstacles=obstacles,
                )

        value_grid = np.clip(value_grid, 0.0, 1.0)
        candidates = cls._score_candidates(candidates, value_grid, top_fraction)
        return cls(value_grid, candidates=candidates, pocket_mask_radius=pocket_mask_radius)

    @classmethod
    def from_grid_evaluator(
            cls,
            evaluator,
            candidates,
            target_ball_pos,
            target_pocket_id,
            obstacles=None,
            m=100,
            n=50,
            pocket_mask_radius=1.5 * ChineseTableSpecs.BALL_RADIUS,
            top_fraction=0.05,
    ):
        heatmap = HeatmapTarget(np.zeros((m, n), dtype=float), pocket_mask_radius=pocket_mask_radius)
        cue_X, cue_Y = np.meshgrid(heatmap.x_coords, heatmap.y_coords)
        valid_mask = heatmap.valid_position_mask(cue_X, cue_Y)
        candidates = tuple(candidates)
        value_grid = evaluator(
            cue_X,
            cue_Y,
            target_ball_pos,
            target_pocket_id,
            candidates=candidates,
            obstacles=obstacles,
        )
        value_grid = np.clip(np.where(valid_mask[np.newaxis, :, :], value_grid, 0.0), 0.0, 1.0)
        candidates = cls._score_candidates(candidates, value_grid, top_fraction)
        return cls(value_grid, candidates=candidates, pocket_mask_radius=pocket_mask_radius)

    def select_best_target_heatmap(self, threshold=0.0, top_k=1, top_fraction=0.05):
        selected_indices = self._top_candidate_indices(top_k)
        if len(selected_indices) > 0:
            scores = np.max(self.value_grid[selected_indices], axis=0)
            score = self._mean_top_fraction(scores, top_fraction)
            selected_options = tuple(self.candidates[i] for i in selected_indices)
        elif self.best_next_shot_index is None and self.k == 1:
            scores = self.value_grid[0].copy()
            score = self._mean_top_fraction(scores, top_fraction)
            selected_options = ()
        elif self.best_next_shot_index is None:
            scores = np.zeros((self.m, self.n), dtype=float)
            score = 0.0
            selected_options = ()
        else:
            scores = self.value_grid[self.best_next_shot_index].copy()
            score = self.candidates[self.best_next_shot_index].score
            selected_options = (self.candidates[self.best_next_shot_index],)

        if threshold > 0:
            scores[scores < threshold] = 0.0

        return (
            HeatmapTarget(
                scores,
                table_width=self.table_width,
                table_length=self.table_length,
                pocket_mask_radius=self.pocket_mask_radius,
            ),
            score,
            selected_options,
        )

    def option_heatmap(self, option_index):
        return HeatmapTarget(
            self.value_grid[option_index].copy(),
            table_width=self.table_width,
            table_length=self.table_length,
            pocket_mask_radius=self.pocket_mask_radius,
        )

    def mean_top_fraction(self, fraction):
        if self.best_next_shot_index is None:
            return 0.0
        return self._mean_top_fraction(self.value_grid[self.best_next_shot_index], fraction)

    def _select_best_next_shot_index(self):
        if len(self.candidates) == 0:
            return None

        candidate_scores = [candidate.score for candidate in self.candidates]
        best_idx = int(np.argmax(candidate_scores))
        if candidate_scores[best_idx] <= 0:
            return None
        return best_idx

    def _top_candidate_indices(self, top_k):
        if len(self.candidates) == 0 or top_k <= 0:
            return []
        candidate_scores = np.asarray([candidate.score for candidate in self.candidates], dtype=float)
        positive = np.flatnonzero(candidate_scores > 0)
        if positive.size == 0:
            return []
        selected = positive[np.argsort(candidate_scores[positive])[::-1][:top_k]]
        return [int(i) for i in selected]

    @classmethod
    def _score_candidates(cls, candidates, value_grid, top_fraction):
        return tuple(
            replace(candidate, score=cls._mean_top_fraction(value_grid[i], top_fraction))
            for i, candidate in enumerate(candidates)
        )

    @staticmethod
    def _mean_top_fraction(scores, fraction):
        if fraction <= 0:
            return 0.0
        values = np.asarray(scores, dtype=float)
        values = values[values > 0]
        if values.size == 0:
            return 0.0
        k = max(1, int(np.ceil(values.size * fraction)))
        return float(np.mean(np.sort(values)[-k:]))
