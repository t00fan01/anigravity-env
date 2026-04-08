from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

# --- The Fix is right here! ---
try:
    from models import AnigravityAction, AnigravityObservation
except ImportError:
    from ..models import AnigravityAction, AnigravityObservation
# ------------------------------

class AnigravityEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.task_name = "easy_hover"
        self._setup_task()

    def _setup_task(self):
        self.dt = 0.1
        self.max_thrust = 20.0
        self.max_steps = 20
        self.step_count = 0
        self.velocity = 0.0

        if self.task_name == "easy_hover":
            self.gravity = 9.8
            self.altitude = 0.0
            self.target_altitude = 10.0
        elif self.task_name == "medium_landing":
            self.gravity = 9.8
            self.altitude = 50.0 
            self.target_altitude = 0.0
        else:
            self.gravity = 15.0 
            self.altitude = 0.0
            self.target_altitude = 20.0

    async def reset(self) -> State:
        self._setup_task()
        return await self.state()

    async def step(self, action: AnigravityAction) -> State:
        self.step_count += 1
        thrust = max(0.0, min(1.0, action.thrust_level))
        acceleration = (thrust * self.max_thrust) - self.gravity
        self.velocity += acceleration * self.dt
        self.altitude += self.velocity * self.dt

        if self.altitude <= 0:
            self.altitude = 0.0
            self.velocity = 0.0

        dist = abs(self.target_altitude - self.altitude)
        if dist < 1.0:
            reward = 1.0
        elif dist < 5.0:
            reward = 0.5
        else:
            reward = 0.0

        done = self.step_count >= self.max_steps
        obs = AnigravityObservation(
            altitude=round(self.altitude, 2),
            velocity=round(self.velocity, 2),
            target_altitude=self.target_altitude
        )
        return State(
            observation=obs.model_dump(),
            reward=reward,
            done=done,
            info={"distance": dist}
        )

    async def state(self) -> State:
        obs = AnigravityObservation(
            altitude=round(self.altitude, 2),
            velocity=round(self.velocity, 2),
            target_altitude=self.target_altitude
        )
        return State(observation=obs.model_dump(), reward=0.0, done=False, info={})