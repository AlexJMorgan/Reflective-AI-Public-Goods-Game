from enum import Enum

class AgentActions(Enum):
    MINIMAL = 0
    MODERATE = 1
    GREEDY = 2
    def __lt__(self, other):
        if type(self) == type(other):
            return self.value < other.value
        else:
            raise TypeError(f"AgentActions cannot be compared by < to an object of type {type(other)}")
    def __gt__(self, other):
        if type(self) == type(other):
            return self.value > other.value
        else:
            raise TypeError(f"AgentActions cannot be compared by > to an object of type {type(other)}")