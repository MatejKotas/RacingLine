from dataclasses import dataclass

class Car(dataclass):
    name: str
    power: float
    braking: float
    cornering: float

base_car = Car("Generic", 100.0, 1.0, 1.0)
