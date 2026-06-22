from dataclasses import dataclass
import numpy as np


@dataclass(order=True)
class ShotAction:
    phi: float = 0.0
    V0: float = 0.0
    a: float = 0.0
    b: float = 0.0
    route_type: str = "direct"  # “direct" | "bank"

    def as_array(self):
        return np.asarray((self.phi, self.V0, self.a, self.b), dtype=float)

    def __repr__(self):
        return (f"ShotAction(phi={self.phi:.2f}, V0={self.V0:.1f}, a={self.a:.1f}, b={self.b:.1f}, "
                f"route_type={self.route_type})")

    def __eq__(self, other):
        if not isinstance(other, ShotAction):
            return False
        return (self.phi, self.V0, self.a, self.b) == (other.phi, other.V0, other.a, other.b)

    def rough_equal(self, other):
        if not isinstance(other, ShotAction):
            return False
        return (
            (-2 <= self.phi - other.phi <= 2) and
            (-0.5 < self.V0 - other.V0 < 0.5) and
            (-0.15 < self.a - other.a < 0.15) and
            (-0.15 < self.b - other.b < 0.15)
        )

    @staticmethod
    def pretty_print_policy(policy):
        if isinstance(policy, ShotAction):
            phi, V0, a, b = policy.as_array()
        elif isinstance(policy, np.ndarray) or isinstance(policy, list) or isinstance(policy, tuple):
            phi, V0, a, b = policy
        else:
            raise TypeError(f"policy type {type(policy)} is not supported")
        print(f"phi: {phi}, V0: {V0}, a: {a}, b: {b}")

    @staticmethod
    def is_spin_possible(a, b, limit=0.650):
        return a * a + b * b < limit * limit

    @staticmethod
    def valid_spin_mask(a_b, limit=0.650):
        return np.sum(a_b * a_b, axis=1) < limit * limit

    @staticmethod
    def is_V0_possible(V0, low=0.25, high=5.0):
        return low <= V0 <= high

    @staticmethod
    def valid_V0_mask(V0, low=0.25, high=5.0):
        return (V0 >= low) & (V0 <= high)