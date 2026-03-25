from marl_models.base_model import MARLModel, ExperienceBatch
import config
import numpy as np


class StaticModel(MARLModel):
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)
        self.static_positions: list[np.ndarray] = self._generate_dynamic_grid(num_agents)

    def _generate_dynamic_grid(self, num_agents: int) -> list[np.ndarray]:
        """Dynamically calculates maximum-spread formations, keeping coverage circles strictly inside the map."""

        boundary_gap: float = config.UAV_COVERAGE_RADIUS  # The circle will just touch the wall!
        safe_width: float = config.AREA_WIDTH - (2 * boundary_gap)
        safe_height: float = config.AREA_HEIGHT - (2 * boundary_gap)
        min_x: float = boundary_gap
        min_y: float = boundary_gap

        max_in_row: int = int(np.ceil(np.sqrt(num_agents)))
        min_required_size: float = (config.MIN_UAV_SEPARATION * (max_in_row - 1)) + (2 * boundary_gap)

        if config.AREA_WIDTH < min_required_size or config.AREA_HEIGHT < min_required_size:
            print(f"\n⚠️ WARNING: Map size ({config.AREA_WIDTH}x{config.AREA_HEIGHT}) is too small for {num_agents} UAVs.")

        num_rows: int = max(1, int(np.round(np.sqrt(num_agents))))
        base_count: int = num_agents // num_rows
        remainder: int = num_agents % num_rows

        row_counts: list[int] = [base_count] * num_rows
        left: int = (num_rows - remainder) // 2
        for i in range(remainder):
            row_counts[left + i] += 1

        positions: list[np.ndarray] = []
        y_coords: list[float] = []

        if num_rows == 1:
            y_coords = [min_y + safe_height / 2.0]
        else:
            y_step: float = safe_height / (num_rows - 1)
            y_coords = [min_y + r * y_step for r in range(num_rows)]

        for r, count_in_row in enumerate(row_counts):
            y = y_coords[r]
            if count_in_row == 1:
                x = min_x + safe_width / 2.0
                positions.append(np.array([x, y], dtype=np.float32))
            else:
                x_step: float = safe_width / (count_in_row - 1)
                for c in range(count_in_row):
                    x = min_x + c * x_step
                    positions.append(np.array([x, y], dtype=np.float32))

        return positions

    def select_actions(self, observations: np.ndarray, exploration: bool = True) -> np.ndarray:
        # Action is exactly [0.0, 0.0] for all UAVs : they hover statically.
        return np.zeros((self.num_agents, self.action_dim), dtype=np.float32)

    def update(self, batch: ExperienceBatch) -> dict:
        return {}  # Does not learn, return empty losses dict.

    def reset(self) -> None:
        pass

    def save(self, directory: str) -> None:
        pass

    def load(self, directory: str) -> None:
        pass
