from pydantic import BaseModel

class AnigravityAction(BaseModel):
    thrust_level: float

class AnigravityObservation(BaseModel):
    altitude: float
    velocity: float
    target_altitude: float
    distance_to_target: float
    fuel_remaining: float
