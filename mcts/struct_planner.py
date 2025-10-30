"""
COL-MCTS
Clustered Open-Loop MCTS

Node is a state s^t representing a sequence of actions a_0:t-1
Edge is an action a_i^t representing a social lever
Associated with each edge are persuader and target utterances,
which we cluster into groups.

Persuader u_{a,k}^t = utterance conditioned on a, in cluster k, at step t
Target    v_m^t = utterance resulting from a persuader response, in cluster m
Transition matrix \pi_{k->m} records num times cluster k led to cluster m
Q_k = \sum_m \pi_{k->m} Q_{k->m}  weighted average reward for cluster k
Q(s,a) = max_k Q_k  reward of best scoring k
"""

from logging import Logger
import random
from typing import List, Tuple

from .mcts_node import OLNode, ConversationState
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

    def plan_conversation(self, initial_state: ConversationState):
        """Execute COL-MCTS search"""

        # note: `initial_state` is the realized conversation up to the
        # point of this tree search call

        self._log(f"\nStarting conversation planning:")
        self._log(f"{"\n\n".join(initial_state.get_annotated_messages())}\n")

        root = OLNode(parent=None, lever=None, p=1./len(self.levers), depth=0)

        for i in range(self.num_simulations):

            # -- SELECTION ---
            # traverse tree to terminal/leaf using selection criteria
            current, path = self._select(root)

            # -- EXPANSION ---
            # if terminal: _expand returns `current`
            # else (leaf): _expand returns a new child
            expanded_node = self._expand(current)
            path.append(expanded_node)

            # -- ROLLOUT ---
            # simulate a conversation along this path. at each edge:
            # select persuader cluster-conditioning and generate,
            # generate target response,
            # update clusters
            state = initial_state.get_deep_copy()
            rec = self._rollout(path, state)

            # -- BACKPROP --
            self._backprop(path, rec)

        # TODO:
        results = []
        for child in root.children:
            results.append((
                child.get_best_candidate(),
                child.get_reward(),
                child.lever,
            ))

    def _select(self, node):
        """Select path through tree using policy"""

        path = [node]

        # so long as we are not at a leaf (leaf = not fully expanded) 
        # and not terminal (= max depth) get "best" child (UCT criteria)
        while (
            not node.is_leaf(self.branching_factor) and 
            not node.is_terminal(self.max_depth)
        ):
            node = node.select_best_child(self.exploration_constant)
            path.append(node)

        # node is either a leaf (not fully expanded) or terminal    
        return node, path

    def _expand(self, node):
        """Expand node by selecting an un-tried lever"""

        # first figure out what levers exist among children of `node`
        existing_levers = [child.lever for child in node.children]

        self._log(f"> Existing levers in children: {existing_levers}")

        # select a new lever randomly from remaining
        remaining_levers = set(self.levers) - set(existing_levers)
        lever = random.choice(list(remaining_levers))

        # construct the new child node
        child = node.add_child(lever=lever)

        return child

    def _rollout(self, path, state):
        """Simulate a conversation along this path"""

        # keep a record of (k, m, r_t) parallel to this path
        rec = []

        # iterate down this path of actions
        for node in path:
            # pick persuader cluster centroid (possibly none)
            persuader_centroid, _ = node.select_best_persuader_response()

            # generate a new response
            # if persuader_centroid=None, get_response ignores it
            persuader_response = self.agents[1].get_response(
                state, lever=node.lever, conditioning=persuader_centroid
            )
            state.add_message(persuader_response)

            # get target reply
            target_response = self.agents[0].get_response(state)
            state.add_message(target_response)

            # add this pair to the node and get its cluster assignments and score
            k, m, r = node.add_response_pair(persuader_response, target_response)

            # update record
            rec.append((k, m, r))

        return rec
    
    def _backprop(self, path, rec):
        """Backprop rewards from rec (k, m, r_t) through path"""

        G = 0.0
        for node, (k, m, rt) in zip(reversed(path), reversed(rec)):
            pass

    
    # ---------------
    #    LOGGING
    # ---------------
    
    def _log(self, message: str):
        self.logger.info(message)

    def _response_preview(self, response: str, max_length: int = 70) -> str:
        """Shortens a response to `max_length`"""
        preview = (
            response[:max_length] + "..." 
            if response is not None and len(response) > max_length 
            else response
        )
        return preview    
