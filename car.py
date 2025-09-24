from dataclasses import dataclass

@dataclass
class Car:
    name: str
    acceleration: float # Measured in m/s^2
    braking: float # Measured in m/s^2
    cornering: float # Measured in m/s^2

base_car = Car("Generic", 6.0, 10.0, 10.0)
inf_car = Car("No constraints", 1000, 1000, 1000)