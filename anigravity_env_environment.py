import random
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State
from models import AnigravityAction, AnigravityObservation

class AnigravityEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.task_name = "easy_hover"
        self._setup_task()

    def _setup_task(self):
        self.dt = 0.1
        self.max_thrust = 20.0
        self.max_steps = 30
        self.step_count = 0
        self.velocity = 0.0
        self.fuel = 100.0

        if self.task_name == "easy_hover":
            self.gravity = 9.8
            self.altitude = 0.0
            self.target_altitude = 10.0
        elif self.task_name == "medium_landing":
            self.gravity = 9.8
            self.altitude = 50.0 
            self.target_altitude = 0.0
        else: # hard_takeoff
            self.gravity = 15.0 
            self.altitude = 0.0
            self.target_altitude = 20.0

    def reset(self) -> State:
        self._setup_task()
        return self.state()

    def step(self, action: AnigravityAction) -> State:
        self.step_count += 1
        
        thrust = max(0.0, min(1.0, action.thrust_level))
        if self.fuel <= 0:
            thrust = 0.0
        else:
            self.fuel -= (thrust * 3.0)

        wind_noise = random.uniform(-0.5, 0.5)
        acceleration = (thrust * self.max_thrust) - self.gravity + wind_noise
        
        self.velocity += acceleration * self.dt
        self.altitude += self.velocity * self.dt

        if self.altitude <= 0:
            self.altitude = 0.0
            if self.velocity < -5.0:
                self.velocity = 0.0 # Hard crash
            else:
                self.velocity = 0.0 # Soft landing

        done = self.step_count >= self.max_steps
        return self.state(reward=0.0, done=done)

    def state(self, reward=0.0, done=False) -> State:
        dist = abs(self.target_altitude - self.altitude)
        obs = AnigravityObservation(
            altitude=round(self.altitude, 2),
            velocity=round(self.velocity, 2),
            target_altitude=self.target_altitude,
            distance_to_target=round(dist, 2),
            fuel_remaining=max(0.0, round(self.fuel, 2))
        )
        return State(observation=obs.model_dump(), reward=reward, done=done, info={})

# --- THE CRASH-PROOF GRADERS ---
# Using *args and try/except guarantees it never throws an error 
# and ALWAYS returns strictly between 0.01 and 0.99.

def grade_easy_hover(*args, **kwargs) -> float:
    try:
        trajectory = args[0] if args else kwargs.get('trajectory', [])
        if not trajectory: return 0.01
        score = 0.0
        for state in trajectory:
            # Safely extract observation whether it's an object or dictionary
            obs = state.get('observation', {}) if isinstance(state, dict) else getattr(state, 'observation', {})
            if isinstance(obs, dict):
                alt = obs.get('altitude', 0.0)
                targ = obs.get('target_altitude', 10.0)
                if abs(targ - alt) <= 2.0:
                    score += 1.0
        
        raw_score = score / len(trajectory) if len(trajectory) > 0 else 0.0
        return float(max(0.01, min(0.99, raw_score)))
    except Exception as e:
        print(f"Grader fail-safe triggered: {e}")
        return 0.02

def grade_medium_landing(*args, **kwargs) -> float:
    try:
        trajectory = args[0] if args else kwargs.get('trajectory', [])
        if not trajectory: return 0.01
        crashed = False
        landed = False
        for state in trajectory:
            obs = state.get('observation', {}) if isinstance(state, dict) else getattr(state, 'observation', {})
            if isinstance(obs, dict):
                alt = obs.get('altitude', 50.0)
                vel = obs.get('velocity', 0.0)
                if alt <= 0.5:
                    landed = True
                    if vel < -5.0:
                        crashed = True
        
        if crashed: return 0.15
        if landed: return 0.95
        return 0.05
    except Exception:
        return 0.02

def grade_hard_takeoff(*args, **kwargs) -> float:
    try:
        trajectory = args[0] if args else kwargs.get('trajectory', [])
        if not trajectory: return 0.01
        last_state = trajectory[-1]
        obs = last_state.get('observation', {}) if isinstance(last_state, dict) else getattr(last_state, 'observation', {})
        
        if isinstance(obs, dict):
            dist = abs(obs.get('target_altitude', 20.0) - obs.get('altitude', 0.0))
            fuel = obs.get('fuel_remaining', 0.0)
            
            if dist <= 3.0:
                raw_score = 0.7 + (0.29 * (fuel / 100.0))
                return float(max(0.01, min(0.99, raw_score)))
        return 0.05
    except Exception:
        return 0.02
