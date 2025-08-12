"""
Main MCTS Conversation Planner implementation.
Uses Monte Carlo Tree Search with UCT policy to select optimal LLM responses.
"""

import logging
from typing import List, Tuple

from .mcts_node import MCTSNode, ConversationState
from .reward_functions import RewardFunction


class ConversationPlanner:
    """
    Monte Carlo Tree Search planner for conversation optimization.
    
    Uses UCT (Upper Confidence Tree) policy to explore conversation trees
    and select the Agent 2 response that maximizes expected reward.
    """
    
    def __init__(self, 
            agents: dict = None,
            reward_function: RewardFunction = None,
            max_depth: int = 3,
            branching_factor: int = 3,
            rollout_depth: int = 5,
            num_simulations: int = 5,
            exploration_constant: float = 1.414,
            dname: str = "test"
        ):
        """
        Initialize conversation planner.
        
        Args:
            agents: dict of Agents
            reward_function: Function to calculate conversation rewards
            max_depth: Maximum depth of search tree
            branching_factor: Max branching at each child
            rollout_depth: max number of turns post-expansion
            num_simulations: Number of MCTS simulations to run
            exploration_constant: UCT exploration parameter
            dname: directory name to save records
        """
        self.agents = agents
        self.reward_function = reward_function
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.rollout_depth = rollout_depth
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant

        # will store records of all sims
        self.records = {}
        
        # Initialize logging
        self.logger = logging.getLogger('MCTS_ConversationPlanner')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def get_records(self) -> dict:
        return self.records
            
    def plan_conversation(self, 
            initial_state: ConversationState, 
            num_candidates: int = 5,
        ) -> List[Tuple[str, float]]:
        """
        Plan conversation by evaluating multiple candidate responses.
        """
        
        # Generate candidate responses from Agent 2
        self._log(f"Generating {num_candidates} candidate responses...")
        candidates = []
        for i in range(num_candidates):
            cand = self.agents[1].get_response(initial_state)
            self._log(f"  Candidate {i}: {self._response_preview(cand)}")
            candidates.append(cand)
        
        self._log(f"\nEvaluating {len(candidates)} candidates with MCTS...")
        
        # Evaluate each candidate using MCTS
        results = []
        for i, candidate in enumerate(candidates):
            self._log(f"=== CANDIDATE {i+1}/{len(candidates)} ===")
                        
            # run MCTS for this candidate
            score, recs = self._evaluate_candidate(initial_state, candidate)
            results.append((candidate, score))

            # save records
            self.records[f"candidate_{i}"] = recs

            self._log(f"\nRESULT ({i+1}): {score}\n")
        
        return results
    
    def _evaluate_candidate(self, 
            initial_state: ConversationState, 
            candidate_response: str
        ) -> Tuple[float, List[dict]]:
        """
        Evaluate a single candidate response using MCTS.
        
        Args:
            initial_state: Initial set of messages
            candidate_response: Agent 2 candidate response to evaluate
            
        Returns:
            Average reward score for this candidate
        """
        # Create state with the candidate response
        state = initial_state.add_message(candidate_response)
        
        # Create root node
        root = MCTSNode(state, parent=None)
        
        # Run MCTS simulations
        records = []
        records.append({'initial_state': state.messages})
        self._log(f"Running {self.num_simulations} MCTS simulations...")
        for simulation in range(self.num_simulations):
            self._log(f" Simulation {simulation + 1}")
            
            record = self._run_simulation(root)
            records.append(record)

            self._log_tree_stats(root, simulation + 1)
        
        # Return average reward
        final_score = root.get_average_reward()
        
        self._log_final_tree_analysis(root)

        return final_score, records
    
    def _run_simulation(self, root: MCTSNode) -> float:
        """Run a single MCTS simulation from the root node."""
        
        record = {}
        record["root"] = root

        # --- Selection phase --- 
        # traverse tree to terminal/leaf using UCT
        current, path = self._select(root)
        self._log_selection_path(path)
        record["select"] = [str(p) for p in path]

        # --- Expansion phase --- 
        # If terminal: _expand returns `current`
        # Else (must be leaf): _expand returns a new child
        expanded_node = self._expand(current)
        self._log_expansion(expanded_node)
        record["expand"] = str(expanded_node)
        
        # --- Rollout phase --- 
        # random rollout from this node
        reward, rollout_state = self._rollout(expanded_node)
        self._log_rollout_result(rollout_state, reward)
        record["rollout"] = rollout_state.messages
        record["reward"] = reward
        
        # --- Backpropagation phase ---
        # update node values starting with the expanded node
        expanded_node.backpropagate(reward)

        return record
            
    def _select(self, node: MCTSNode) -> Tuple[MCTSNode, List[MCTSNode]]:
        """Select path through tree using UCT policy."""
        
        # initialize path
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
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding a new child."""

        # if at max tree depth, don't add child and we'll rollout from here
        if node.is_terminal(self.max_depth):
            return node
        
        # create new state for child
        current_state = node.state
        agent = self.agents[current_state.current_turn]  # current_turn=awaiting
        next_message = agent.get_response(current_state)
        new_state = current_state.add_message(next_message)
        
        # add child node
        child = node.add_child(action=next_message, state=new_state)

        return child
    
    def _rollout(self, node: MCTSNode) -> Tuple[float, List[str]]:
        """Rollout random conversation from current node and score"""
        
        # simulate self.rollout_depth rounds of dialogue
        sim_state = ConversationState(messages=node.state.messages.copy())
        for _ in range(self.rollout_depth):
            agent = self.agents[sim_state.current_turn]
            next_message = agent.get_response(sim_state)
            sim_state = sim_state.add_message(next_message)
        
        # score
        reward = self.reward_function.calculate_reward(sim_state)
        return reward, sim_state


    # ----------------------
    # Logging helper methods
    # ----------------------
    
    def _log(self, message: str):
        """Log an info message."""
        self.logger.info(message)

    def _log_selection_path(self, path: List[MCTSNode]):
        """Log the selection path through the tree."""

        self.logger.info("  Selection Path:")
        for i, node in enumerate(path):
            depth_info = f"depth={node.state.depth}"
            visits_info = f"visits={node.visits}"
            reward_info = f"avg_reward={node.get_average_reward():.4f}"
            children_info = f"children={len(node.children)}"
            
            if i == 0:
                self.logger.info(f"    Root -> {depth_info}, {visits_info}, {reward_info}, {children_info}")
            else:
                action_preview = self._response_preview(node.action)
                self.logger.info(f"    └─> \"{action_preview}\" -> {depth_info}, {visits_info}, {reward_info}, {children_info}")

    def _log_expansion(self, expanded_node: MCTSNode):
        """Log node expansion."""
        action_preview = self._response_preview(expanded_node.action)
        self.logger.info(
            f"  Expanded: \"{action_preview}\" "
            f"(depth={expanded_node.state.depth})"
        )

    def _log_rollout_result(self, rollout_state: ConversationState, reward: float):
        """Log simulation rollout and result."""
        
        # NOTE: rollout_state contains the ENTIRE conversation, from initial
        # prompt.  We may want to just show the rollout portion
        self.logger.info("  Simulation Rollout:")
        for step in rollout_state.get_annotated_messages():
            step_preview = self._response_preview(step)
            self.logger.info(f"    {step_preview}")
        self.logger.info(f"  Simulation Reward: {reward:.6f}")
    
    def _log_tree_stats(self, root: MCTSNode, simulation_num: int):
        """Log tree statistics."""
        total_nodes = self._count_nodes(root)
        self.logger.info(
            f"  After {simulation_num} simulations: {total_nodes} nodes, "
            f"avg_reward={root.get_average_reward():.6f}"
        )
    
    def _log_final_tree_analysis(self, root: MCTSNode):
        """Log final tree analysis."""

        self.logger.info(f"\nFinal Tree Analysis:")
        self.logger.info(f"  Root visits: {root.visits}")
        self.logger.info(f"  Root average reward: {root.get_average_reward():.6f}")
        self.logger.info(f"  Total nodes in tree: {self._count_nodes(root)}")
        
        if root.children:
            self.logger.info(f"  Root children ({len(root.children)}):")
            # Sort children by average reward
            sorted_children = sorted(root.children, key=lambda x: x.get_average_reward(), reverse=True)
            for i, child in enumerate(sorted_children[:5]):  # Show top 5
                action_preview = self._response_preview(child.action)
                self.logger.info(
                    f"    {i+1}. \"{action_preview}\" -> visits={child.visits}, "
                    f"avg_reward={child.get_average_reward():.6f}"
                )
    
    def _response_preview(self, response: str, max_length: int = 70) -> str:
        """Shortens a response to `max_length`"""
        preview = (
            response[:max_length] + "..." 
            if response is not None and len(response) > max_length 
            else response
        )
        return preview

    def _count_nodes(self, node: MCTSNode) -> int:
        """Count total nodes in subtree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
