from logging import Logger
import random
from typing import List, Tuple

from .mcts_node import LeverNode, ConversationState
from .reward_functions import RewardFunction


class StructPlanner:
     
    def __init__(self, 
            agents: dict = None,
            reward_function: RewardFunction = None,
            max_depth: int = 3,
            branching_factor: int = 3,
            generations_per_node: int = 5,
            num_simulations: int = 5,
            exploration_constant: float = 1.414,
            levers: List[str] = None,
            logger: Logger = None,
        ):
        self.agents = agents
        self.reward_function = reward_function
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.generations_per_node = generations_per_node
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.levers = levers
        self.logger = logger

        # will store records of all sims
        self.records = {}
