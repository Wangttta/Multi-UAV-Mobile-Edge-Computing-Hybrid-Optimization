import config
import numpy as np


class UE:
    all_ids: np.ndarray
    global_ranks: np.ndarray
    id_to_rank_map: dict[int, int]
    global_probs: np.ndarray
    hotspot_centers: list[np.ndarray]

    @classmethod
    def initialize_ue_class(cls) -> None:
        cls.all_ids = np.arange(config.NUM_FILES)  # Assume IDs 0 to NUM_SERVICES-1 are Services, rest are Contents
        cls.global_ranks = np.arange(1, config.NUM_FILES + 1)
        np.random.shuffle(cls.global_ranks)  # Currently random ranks assigned
        cls.id_to_rank_map = dict(zip(cls.all_ids, cls.global_ranks))  # Mapping from ID to rank
        zipf_denom: float = np.sum(1 / cls.global_ranks**config.ZIPF_BETA)
        cls.global_probs = (1 / cls.global_ranks**config.ZIPF_BETA) / zipf_denom

        if getattr(config, "USE_HOTSPOTS", False):
            cls.generate_hotspots()

    @classmethod
    def generate_hotspots(cls) -> None:
        """Randomizes the locations of the hotspots across the map."""
        cls.hotspot_centers = []
        max_retries = 100  # Safety limit for rejection sampling

        for _ in range(config.NUM_HOTSPOTS):
            valid: bool = False
            new_center: np.ndarray = np.zeros(2, dtype=np.float32)
            retries: int = 0
            while not valid and retries < max_retries:
                hx: float = np.random.uniform(config.HOTSPOT_RADIUS, config.AREA_WIDTH - config.HOTSPOT_RADIUS)
                hy: float = np.random.uniform(config.HOTSPOT_RADIUS, config.AREA_HEIGHT - config.HOTSPOT_RADIUS)
                new_center = np.array([hx, hy], dtype=np.float32)

                if not cls.hotspot_centers:
                    valid = True
                else:
                    distances: np.ndarray = np.linalg.norm(np.array(cls.hotspot_centers) - new_center, axis=1)
                    if np.min(distances) > config.HOTSPOT_SEPARATION:
                        valid = True
                retries += 1

            cls.hotspot_centers.append(new_center)

    def __init__(self, ue_id: int) -> None:
        self.id: int = ue_id
        self.pos: np.ndarray = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT), 0.0], dtype=np.float32)
        self.is_hotspot_user = getattr(config, "USE_HOTSPOTS", False) and (self.id < config.NUM_UES * getattr(config, "HOTSPOT_UE_PROB", 0.0))
        if self.is_hotspot_user:
            self.pos[:2] = self._get_position_in_hotspot()

        self.battery_level: float = np.random.uniform(0.6, 1.0) * config.UE_BATTERY_CAPACITY  # Start at capacity between 60% to 100%

        self.current_request: tuple[int, int, int] = (0, 0, 0)  # Request : (req_type, req_size, req_id)
        self.latency_current_request: float = 0.0  # Latency for the current request
        self.assigned: bool = False

        # Random Waypoint Model
        self._waypoint: np.ndarray
        self._wait_time: int
        self._set_new_waypoint()  # Initialize first waypoint

        # Fairness Tracking
        self._successful_requests: int = 0
        self.service_coverage: float = 0.0

    def update_position(self) -> None:
        """Updates the UE's position for one time slot as per the Random Waypoint model."""
        if self._wait_time > 0:
            self._wait_time -= 1
            return

        direction_vec: np.ndarray = self._waypoint - self.pos[:2]
        distance_to_waypoint: float = float(np.linalg.norm(direction_vec))

        if config.UE_MAX_DIST >= distance_to_waypoint:  # Reached the waypoint
            self.pos[:2] = self._waypoint
            self._set_new_waypoint()
        else:  # Move towards the waypoint
            move_vector = (direction_vec / distance_to_waypoint) * config.UE_MAX_DIST
            self.pos[:2] += move_vector

    def _get_position_in_hotspot(self) -> np.ndarray:
        """Generates a random position strictly within this UE's assigned hotspot."""
        angle: float = np.random.uniform(0, 2 * np.pi)
        r: float = config.HOTSPOT_RADIUS * np.sqrt(np.random.uniform(0, 1))
        offset: np.ndarray = r * np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
        center: np.ndarray = UE.hotspot_centers[self.id % config.NUM_HOTSPOTS]
        pos: np.ndarray = np.clip(center + offset, [0, 0], [config.AREA_WIDTH, config.AREA_HEIGHT])
        return pos.astype(np.float32)

    def generate_request(self) -> None:
        """Generates a new request tuple for the current time slot."""

        # Check for Emergency Energy Request
        if self.battery_level < config.UE_CRITICAL_THRESHOLD:
            self.current_request = (2, 0, 0)
            self.latency_current_request = 0.0
            self.assigned = False
            return

        req_id: int = np.random.choice(UE.all_ids, p=UE.global_probs)
        req_type: int = 0 if req_id < config.NUM_SERVICES else 1
        req_size: int = np.random.randint(config.MIN_INPUT_SIZE, config.MAX_INPUT_SIZE) if req_type == 0 else 0
        self.current_request = (req_type, req_size, req_id)
        self.latency_current_request = 0.0
        self.assigned = False

    def update_service_coverage(self, current_time_step_t: int) -> None:
        """Updates the fairness metric based on service outcome in the current slot."""
        if self.assigned and self.latency_current_request <= config.TIME_SLOT_DURATION:
            self._successful_requests += 1

        assert current_time_step_t > 0
        self.service_coverage = self._successful_requests / current_time_step_t

    def _set_new_waypoint(self):
        """Set a new destination, speed, and wait time as per the Random Waypoint model."""
        # If hotspots are active, the new waypoint MUST also be inside the hotspot!
        if self.is_hotspot_user:
            self._waypoint = self._get_position_in_hotspot()
        else:
            self._waypoint = np.array([np.random.uniform(0, config.AREA_WIDTH), np.random.uniform(0, config.AREA_HEIGHT)], dtype=np.float32)

        self._wait_time = np.random.randint(0, config.UE_MAX_WAIT_TIME + 1)

    def update_battery(self, harv_energy: float, ue_transmit_time: float) -> None:
        """Updates battery level based on consumption and harvesting."""
        consumed_energy: float = config.UE_STATIC_POWER * config.TIME_SLOT_DURATION
        consumed_energy += config.TRANSMIT_POWER * ue_transmit_time
        self.battery_level = min(config.UE_BATTERY_CAPACITY, self.battery_level - consumed_energy + harv_energy)
        self.battery_level = max(0.0, self.battery_level)
