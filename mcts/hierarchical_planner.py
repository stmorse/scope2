"""
MCTS *Hierarchical* Conversation planner
At each node, expansion involves first selecting a social dimension,
then generating k candidates and responses conditioned on that dimension,
scoring, and selecting greedily the best one
(no rollout)
"""

from logging import Logger
import random
from typing import List, Tuple

from .mcts_node import LeverNode, ConversationState
from .reward_functions import RewardFunction


class HierarchicalPlanner:
     
    def __init__(self, 
            agents: dict = None,
            reward_function: RewardFunction = None,
            max_depth: int = 3,
            branching_factor: int = 3,
            generations_per_node: int = 5,
            # rollout_depth: int = 5,
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
        # self.rollout_depth = rollout_depth
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.levers = levers
        self.logger = logger

        # will store records of all sims
        self.records = {}

    def get_records(self) -> dict:
        return self.records

    def plan_conversation(self, initial_state: ConversationState):
        """Performs (hierarchical) MCTS over self.levers"""

        self._log(f"\nStarting convo:") 
        self._log(f"{"\n\n".join(initial_state.get_annotated_messages())}\n")

        # one root for all sims, contains a state ending in a tgt message
        root = LeverNode(initial_state, parent=None, depth=0)

        self._log(f"Starting {self.num_simulations} simulations ...")
        for i in range(self.num_simulations):
            self._log(f"\n*** SIMULATION {i+1}/{self.num_simulations} ***\n")

            self.records[f"sim_{i}"] = []

            # --- Selection phase --- 
            # traverse tree to terminal/leaf using UCT
            current, path = self._select(root)
            self._log(f"Selection path:")
            self._log(f"{'\n'.join([str(p) for p in path])}")

            # --- Expansion phase --- 
            # If terminal: _expand returns `current`
            # Else (must be leaf): _expand returns a new child
            expanded_node, score = self._expand(current)

            # --- (ROLLOUT) ---
            # TODO

            # --- BACKPROP ---
            expanded_node.backpropagate(score)

            self._log(f"Expanded node:\n{str(expanded_node)}\n")

            self.records[f"sim_{i}"].append({
                "path": [str(p) for p in path],
                "selected": str(current),
                "expanded": str(expanded_node),
                "messages": [expanded_node.state.messages],
                "score": score
            })

        # get results and scores 
        results = []
        for child in root.children:
            results.append((
                child.state.messages[-2], 
                child.get_average_reward(),
                child.lever,
            ))

        return results, root


    def _select(self, node):
        """Select path through tree using MCTSNode (UCT) policy."""

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
        """
        Expand node by selecting an un-tried lever and picking best response
        """

        self._log("\nEXPANDING\n")

        self._log(f"> Node:\n{str(node)}\n")

        # if at max tree depth, don't add child 
        # NOTE: we should not hit this condition, we check against it 
        if node.is_terminal(self.max_depth):
            print(f"> Node is terminal, no expansion")
            return node, node.get_average_reward()

        # first figure out what levers exist among children of `node`
        existing_levers = [child.lever for child in node.children]

        self._log(f"> Existing levers in children: {existing_levers}")

        # select a new lever randomly from remaining
        remaining_levers = set(self.levers) - set(existing_levers)
        lever = random.choice(list(remaining_levers))

        self._log(f"> Expanding with lever: {lever}")

        self._log("\nCONVERSATION STATE:")
        self._log("\n".join(node.state.get_annotated_messages()))
        self._log("---")

        # simulate conditioned response + tgt reply
        responses = []
        tgt_replies = []
        scores = []
        best_score_index = -1
        for k in range(self.generations_per_node):
            self._log(f"\n> GENERATION {k+1}/{self.generations_per_node}")
            # create a copy of state for this generation
            temp_state = node.state.get_deep_copy()

            # get conditioned response
            response = self.agents[1].get_response(temp_state, lever=lever)
            responses.append(response)

            # add to state and get reply
            temp_state = temp_state.add_message(response)
            tgt_reply = self.agents[0].get_response(temp_state)
            tgt_replies.append(tgt_reply)

            # add to state and score state
            temp_state = temp_state.add_message(tgt_reply)
            score = self.reward_function.calculate_reward(temp_state)
            scores.append(score)

            self._log(f"> Agent 1 response: {response}")
            self._log(f"> Agent 0 response: {tgt_reply}")
            self._log(f"> Score: {score:.3f}")

            # keep track of greedy pick
            if best_score_index < 0 or score > scores[best_score_index]:
                best_score_index = k

        self._log("\n> EXPANSION RESULT:")
        self._log(f"> Best A1 reply: {responses[best_score_index]}")
        self._log(f"> A0 reply: {tgt_replies[best_score_index]}")
        self._log(f"> (Score: {scores[best_score_index]})\n")

        # construct the new child node
        new_state = node.state.add_message(responses[best_score_index])
        new_state = new_state.add_message(tgt_replies[best_score_index])
        child = node.add_child(
            state=new_state,
            action=node.state.messages[-1], 
            lever=lever,
            generations=list(zip(responses, tgt_replies)),
        )

        return child, scores[best_score_index]

    
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