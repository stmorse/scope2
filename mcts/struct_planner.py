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
from .reward_functions import RewardFunction, NLIReward, TopicReward


class StructPlanner:
     
    def __init__(self, 
            agents: dict = None,
            persuader_reward: RewardFunction = None,
            target_reward: RewardFunction = None,
            max_depth: int = 3,
            branching_factor: int = 3,
            generations_per_node: int = 5,
            num_simulations: int = 5,
            exploration_constant: float = 1.414,
            decay: float = 0.9,
            levers: List[str] = None,
            logger: Logger = None,
        ):
        self.agents = agents

        # rewards / embedders
        self.persuader_reward = persuader_reward
        self.target_reward = target_reward

        # tree hyperparams
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.generations_per_node = generations_per_node
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.decay = decay  # reward decay
        self.levers = levers
        
        # logging
        self.logger = logger

        # will store records of all sims
        self.records = {}

    def get_records(self):
        return self.records

    def reset(self):
        self.records = {}

    def plan_conversation(self, initial_state: ConversationState):
        """Execute COL-MCTS search"""

        # note: `initial_state` is the realized conversation up to the
        # point of this tree search call

        self._log(f"\nStarting conversation planning:")
        self._log(f"{"\n\n".join(initial_state.get_annotated_messages())}\n")

        root = OLNode(
            parent=None, lever=None, p=1./len(self.levers), depth=0,
            persuader_reward=self.persuader_reward,
            target_reward=self.target_reward
        )

        for i in range(self.num_simulations):
            self._log(f"\nSIMULATION {i+1} / {self.num_simulations}\n")

            # -- SELECTION ---
            # traverse tree to terminal/leaf using selection criteria
            path = self._select(root)

            # -- EXPANSION ---
            # if terminal: _expand returns `current`
            # else (leaf): _expand returns a new child
            expanded_node = self._expand(path[-1])
            path.append(expanded_node)

            self._log(f"Select/Expand path:")
            self._log(f"{'\n'.join([str(p) for p in path])}")

            # -- ROLLOUT ---
            # simulate a conversation along this path. at each edge:
            # select persuader cluster-conditioning and generate,
            # generate target response,
            # update clusters
            state = initial_state.get_deep_copy()
            rec, state = self._rollout(path, state)

            # -- BACKPROP --
            self._backprop(path, rec)

            self.records[f"sim_{i}"] = {
                "path": [str(p) for p in path],
                "messages": [m for m in state.messages],
                "score": [float(val) for _, _, val in rec],
            }

        results = []
        for child in root.children:
            results.append((
                child.get_best_persuader_candidate(),
                float(child.get_q()),
                child.lever,
            ))

        return results, root

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

        # final node is either a leaf (not fully expanded) or terminal    
        return path

    def _expand(self, node):
        """Expand node by selecting an un-tried lever"""

        # first figure out what levers exist among children of `node`
        existing_levers = [child.lever for child in node.children]

        self._log(f"Existing levers in children: {existing_levers}")

        # select a new lever randomly from remaining
        remaining_levers = set(self.levers) - set(existing_levers)
        lever = random.choice(list(remaining_levers))

        self._log(f"Expanding with lever: {lever}")

        # construct the new child node
        child = node.add_child(lever=lever)

        return child

    def _rollout(self, path, state):
        """
        Simulate a conversation along this path and record the specific
        persuader / target clusters and rewards we see
        """

        self._log("\nRollout...\n")

        # keep a record of (k, m, r_t) parallel to this path
        rec = []

        # this is starting reward
        # TODO: need to fix this
        L_prev = 0

        # iterate down this path of actions
        for node in path:
            print(f"node nkm:\n{node.nkm}")

            # pick persuader cluster centroid (possibly none)
            persuader_centroid, _ = node.select_best_persuader_response()

            self._log(f"Conditioning on: {persuader_centroid}")

            # generate a new response
            # if persuader_centroid=None, get_response ignores it
            persuader_response = self.agents[1].get_response(
                state, lever=node.lever, conditioning=persuader_centroid
            )
            state = state.add_message(persuader_response)

            # get target reply
            target_response = self.agents[0].get_response(state)
            state = state.add_message(target_response)

            # add this pair to the node and get its cluster assignments and score
            k, m, L = node.add_response_pair(state)
            r = L - L_prev

            self._log(f"Response pair:\n{persuader_response}\n{target_response}")
            self._log(f"rec: {(k, m, r)}")

            # update record
            rec.append((k, m, r))
            L_prev = L

        # this list runs t=0,...,T-1
        return rec, state
    
    def _backprop(self, path, rec):
        """Backprop rewards from rec (k, m, r_t) through path"""
        G = 0.0
        for node, (k, m, r) in zip(reversed(path), reversed(rec)):
            # NOTE:
            # at last node (T-1), this gives G=r_{T-1}
            # at second to last, we have G=r_{T-2}+decay*r_{T-1} ...
            # at t, we have G_t = \sum_{u=t}^{T-1} decay^{u-t} r_t
            G = r + self.decay * G
            node.update(k, m, G)

    
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
