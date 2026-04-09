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
        self.fuel = 100.0  # The drone now has a battery!

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
        
        # 1. Fuel Burn Mechanics
        thrust = max(0.0, min(1.0, action.thrust_level))
        if self.fuel <= 0:
            thrust = 0.0  # Out of fuel, engines cut out
        else:
            self.fuel -= (thrust * 3.0)  # Burn fuel based on thrust output

        # 2. Physics with random wind noise
        wind_noise = random.uniform(-0.5, 0.5)
        acceleration = (thrust * self.max_thrust) - self.gravity + wind_noise
        
        self.velocity += acceleration * self.dt
        self.altitude += self.velocity * self.dt

        # 3. Collision Logic
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

# --- THE REAL GRADERS (These determine the final score) ---

def grade_easy_hover(trajectory) -> float:
    # Reward for spending time close to target altitude
    if not trajectory: return 0.0
    score = 0.0
    for state in trajectory:
        obs = state.observation if hasattr(state, 'observation') else state['observation']
        if abs(obs['target_altitude'] - obs['altitude']) <= 2.0:
            score += 1.0
    return min(1.0, score / len(trajectory))

def grade_medium_landing(trajectory) -> float:
    # Must reach altitude 0 safely without crashing at high speed
    if not trajectory: return 0.0
    crashed = False
    landed = False
    for state in trajectory:
        obs = state.observation if hasattr(state, 'observation') else state['observation']
        if obs['altitude'] <= 0.5:
            landed = True
            if obs['velocity'] < -5.0:
                crashed = True # Hit the ground too hard!
    
    if crashed: return 0.0
    if landed: return 1.0
    return 0.0

def grade_hard_takeoff(trajectory) -> float:
    # Reach target altitude and keep fuel efficient
    if not trajectory: return 0.0
    final_obs = trajectory[-1].observation if hasattr(trajectory[-1], 'observation') else trajectory[-1]['observation']
    dist = abs(final_obs['target_altitude'] - final_obs['altitude'])
    
    if dist <= 3.0:
        fuel_bonus = final_obs['fuel_remaining'] / 100.0
        return min(1.0, 0.7 + (0.3 * fuel_bonus))
    return 0.0
