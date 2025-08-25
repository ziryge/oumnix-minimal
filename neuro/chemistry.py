"""
"""
class NeuroState:
    def __init__(self):
        
        self.dopamine = 0.5
        self.serotonin = 0.5
        self.noradrenaline = 0.5

    def update(self, reward: float, surprise: float):
        """
"""
        
        self.dopamine = max(0.0, min(1.0, self.dopamine + 0.1 * reward))
        
        self.serotonin = max(0.0, min(1.0, self.serotonin - 0.05 * surprise))
        
        self.noradrenaline = max(0.0, min(1.0, self.noradrenaline + 0.07 * surprise))

    def temperature(self) -> float:
        """
"""
        return 0.7 + 0.4 * self.dopamine

    def depth_factor(self) -> float:
        """
"""
        return 1.0 - 0.3 * self.serotonin
