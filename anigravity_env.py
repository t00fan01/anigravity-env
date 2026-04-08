from pydantic import BaseModel, Field
from openenv.core import BaseEnv, StepResult, StateResult

# --- 1. What the AI sees ---
class AnigravityObservation(BaseModel):
    altitude: float = Field(0.0, description="Current height in meters")
    velocity: float = Field(0.0, description="Current vertical speed")
    target_altitude: float = Field(10.0, description="Target height to maintain")

# --- 2. What the AI does ---
class AnigravityAction(BaseModel):
    thrust_level: float = Field(0.0, ge=0.0, le=1.0, description="Anti-gravity thrust power (0 to 1)")

# --- 3. The Virtual World Engine ---
class AnigravityEnv(BaseEnv[AnigravityObservation, AnigravityAction]):
    def __init__(self, task_name: str = "easy_hover"):
        super().__init__()
        self.dt = 0.1
        self.max_thrust = 20.0
        self.max_steps = 50
        self.task_name = task_name
        self.setup_task()

    def setup_task(self):
        """Sets up the 3 different difficulty levels required by the Hackathon"""
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
        elif self.task_name == "hard_heavy_payload":
            self.gravity = 15.0 
            self.altitude = 0.0
            self.target_altitude = 20.0
        else:
            self.gravity = 9.8
            self.altitude = 0.0
            self.target_altitude = 10.0

    def reset(self) -> StateResult[AnigravityObservation]:
        self.setup_task()
        return self.state()

    def step(self, action: AnigravityAction) -> StepResult[AnigravityObservation]:
        self.step_count += 1

        # Calculate Physics
        thrust = max(0.0, min(1.0, action.thrust_level))
        acceleration = (thrust * self.max_thrust) - self.gravity
        self.velocity += acceleration * self.dt
        self.altitude += self.velocity * self.dt

        # Stop from falling through the floor
        if self.altitude <= 0:
            self.altitude = 0.0
            self.velocity = 0.0

        # Calculate Reward (Partial progress signals as requested by rubric)
        dist = abs(self.target_altitude - self.altitude)
        if dist < 1.0:
            reward = 1.0  # Perfect hover
        elif dist < 5.0:
            reward = 0.5  # Getting close
        else:
            reward = 0.0  # Too far away

        done = self.step_count >= self.max_steps

        return StepResult(
            observation=self.state().observation,
            reward=reward,
            done=done,
            info={"distance": dist}
        )

    def state(self) -> StateResult[AnigravityObservation]:
        obs = AnigravityObservation(
            altitude=round(self.altitude, 2),
            velocity=round(self.velocity, 2),
            target_altitude=self.target_altitude
        )
        return StateResult(observation=obs)