from marl_models.base_model import MARLModel, ExperienceBatch
import config
import numpy as np


class NearestGreedyModel(MARLModel):
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)

        # Calculate exactly where the UE observations start in the flattened array
        self.ue_start_idx: int = 2 + config.NUM_FILES + config.MAX_UAV_NEIGHBORS * config.NEIGHBOR_OBS_DIM

    def select_actions(self, observations: np.ndarray, exploration: bool = True) -> np.ndarray:
        # In obs, UEs are pre-sorted by distance. Hence, the very first UE block in the observation is the nearest.
        dx: np.ndarray = observations[:, self.ue_start_idx] * config.AREA_WIDTH
        dy: np.ndarray = observations[:, self.ue_start_idx + 1] * config.AREA_HEIGHT
        distances: np.ndarray = np.sqrt(dx**2 + dy**2)

        # Default action: If no UEs are in range (distance is 0 due to zero-padding), do a moderate random walk.
        actions: np.ndarray = np.random.uniform(-0.2, 0.2, size=(self.num_agents, self.action_dim)).astype(np.float32)

        # Masking: Only apply the greedy vector to UAVs that actually see a UE
        mask: np.ndarray = distances > 0.1
        if np.any(mask):
            actions[mask, 0] = dx[mask] / distances[mask]
            actions[mask, 1] = dy[mask] / distances[mask]
        return actions

    def update(self, batch: ExperienceBatch) -> dict:
        return {}  # Does not learn, return empty losses dict.

    def reset(self) -> None:
        pass

    def save(self, directory: str) -> None:
        pass

    def load(self, directory: str) -> None:
        pass
