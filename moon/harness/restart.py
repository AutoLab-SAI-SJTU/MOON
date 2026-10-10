"""What the loop restarts the base policy on after an alarm."""
from typing import Dict


class IdentityRestart:
    """Restart on the original instruction; with a driver, the recovery adapter
    drives for the rest of the episode."""

    def __init__(self, instruction: str, driver=None, role: str = "restart"):
        self.instruction = instruction
        self.driver = driver
        self.role = role

    def propose(self, state) -> Dict:
        if self.driver is not None:
            self.driver.policy_role = self.role
        return {"text": self.instruction, "verdict": "identity"}

