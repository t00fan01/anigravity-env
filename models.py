from pydantic import BaseModel, Field

class AnigravityObservation(BaseModel):
    altitude: float = Field(0.0, description="Current height in meters")
    velocity: float = Field(0.0, description="Current vertical speed")
    target_altitude: float = Field(10.0, description="Target height to maintain")

class AnigravityAction(BaseModel):
    thrust_level: float = Field(0.0, ge=0.0, le=1.0, description="Anti-gravity thrust power (0 to 1)")