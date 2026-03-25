from marl_models.base_model import MARLModel, ExperienceBatch
import config
import numpy as np


class UncoordinatedGreedyModel(MARLModel):
    def __init__(self, model_name: str, num_agents: int, obs_dim: int, action_dim: int, device: str) -> None:
        super().__init__(model_name, num_agents, obs_dim, action_dim, device)

        # The greedy target: The absolute center of the map
        self.target_pos: np.ndarray = np.array([[config.AREA_WIDTH / 2.0, config.AREA_HEIGHT / 2.0]], dtype=np.float32)
        self.area_dims: np.ndarray = np.array([[config.AREA_WIDTH, config.AREA_HEIGHT]], dtype=np.float32)

    def select_actions(self, observations: np.ndarray, exploration: bool = True) -> np.ndarray:
        current_pos: np.ndarray = observations[:, :2] * self.area_dims
        delta: np.ndarray = self.target_pos - current_pos  # Shape: (num_agents, 2)
        distances: np.ndarray = np.linalg.norm(delta, axis=1, keepdims=True)  # Shape: (num_agents, 1)

        # Default action: Random jitter (fighting for the center spot)
        actions: np.ndarray = np.random.uniform(-0.1, 0.1, size=(self.num_agents, self.action_dim)).astype(np.float32)

        # Only apply the directed flight to UAVs farther than 10m from the center
        mask: np.ndarray = (distances > 10.0).flatten()
        if np.any(mask):
            actions[mask] = delta[mask] / distances[mask]  # Unit vector towards center

        return actions

    def update(self, batch: ExperienceBatch) -> dict:
        return {}  # Does not learn, return empty losses dict.

    def reset(self) -> None:
        pass

    def save(self, directory: str) -> None:
        pass

    def load(self, directory: str) -> None:
        pass
