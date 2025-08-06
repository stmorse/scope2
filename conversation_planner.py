"""
Main MCTS Conversation Planner implementation.
Uses Monte Carlo Tree Search with UCT policy to select optimal LLM responses.
"""

import random
import math
import logging
from typing import List, Tuple, Optional
from agent.agent import Agent
# from llm_providers import LLMProvider
from mcts_node import MCTSNode, ConversationState
from reward_functions import RewardFunction


class ConversationPlanner:
    """
    Monte Carlo Tree Search planner for conversation optimization.
    
    Uses UCT (Upper Confidence Tree) policy to explore conversation trees
    and select the Agent 2 response that maximizes expected reward.
    """
    
    def __init__(self, 
            agents: List[Agent],
            reward_function: RewardFunction = None,
            max_depth: int = 3,
            branching_factor: int = 3,
            rollout_depth: int = 5,
            num_simulations: int = 5,
            exploration_constant: float = math.sqrt(2),
            temperature: float = 0.7,
        ):
        """
        Initialize conversation planner.
        
        Args:
            agents: list of Agents
            reward_function: Function to calculate conversation rewards
            max_depth: Maximum depth of search tree
            branching_factor: Max branching at each child
            rollout_depth: max number of turns post-expansion
            num_simulations: Number of MCTS simulations to run
            exploration_constant: UCT exploration parameter
            temperature: Temperature for LLM generation
        """
        self.agents = agents
        self.reward_function = reward_function
        self.max_depth = max_depth
        self.branching_factor = branching_factor
        self.rollout_depth = rollout_depth
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.temperature = temperature
        
        # Initialize logging
        self.logger = logging.getLogger('MCTS_ConversationPlanner')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Statistics tracking
        self.total_simulations_run = 0
        self.total_nodes_created = 0
    
    def plan_conversation(self, 
            initial_prompt: str, 
            num_candidates: int = 5
        ) -> List[Tuple[str, float]]:
        """
        Plan conversation by evaluating multiple Agent 2 candidate responses.
        
        Args:
            initial_prompt: Initial message from Agent 1
            num_candidates: Number of candidate responses to evaluate
            
        Returns:
            List of (candidate_response, score) tuples
        """
        
        # Generate candidate responses from Agent 2
        print(f"Generating {num_candidates} candidate responses...")
        candidates = [
            self.agents[1].get_response(prompt) for _ in range(num_candidates)
        ]
        
        print(f"Evaluating {len(candidates)} candidates with MCTS...")
        
        # Evaluate each candidate using MCTS
        results = []
        for i, candidate in enumerate(candidates):
            self._log_candidate_header(i+1, len(candidates), candidate)
            
            # Reset simulation counter for this candidate
            self.total_simulations_run = 0
                        
            # run MCTS for this candidate
            score = self._evaluate_candidate(initial_prompt, candidate)
            results.append((candidate, score))
            
            self._log_candidate_result(i+1, candidate, score)
        
        return results
    
    def _evaluate_candidate(self, initial_prompt: str, candidate_response: str) -> float:
        """
        Evaluate a single candidate response using MCTS.
        
        Args:
            initial_prompt: Initial Agent 1 message
            candidate_response: Agent 2 candidate response to evaluate
            
        Returns:
            Average reward score for this candidate
        """
        # Create initial state with the candidate response
        initial_state = ConversationState(
            messages=[initial_prompt, candidate_response],
            current_turn=0,  # Next turn is Agent 1
            depth=1
        )
        
        # Create root node
        root = MCTSNode(initial_state, parent=None)
        
        # Run MCTS simulations
        self._log(f"Running {self.num_simulations} MCTS simulations...")
        for simulation in range(self.num_simulations):
            self._log_simulation_header(simulation + 1)
            
            self._run_simulation(root, simulation + 1)

            self._log_tree_stats(root, simulation + 1)
        
        # Return average reward
        final_score = root.get_average_reward()
        
        self._log_final_tree_analysis(root)

        return final_score
    
    def _run_simulation(self, 
            root: MCTSNode, 
            simulation_num: int = 0
        ) -> float:
        """Run a single MCTS simulation from the root node."""
        
        # --- Selection phase --- 
        # traverse tree to terminal/leaf using UCT
        current, path = self._select(root)
        self._log_selection_path(path)
        
        # --- Expansion phase --- 
        # If terminal: _expand returns `current`
        # Else (must be leaf): _expand returns a new child
        expanded_node = self._expand(current)
        self._log_expansion(expanded_node)
        
        # --- Simulation phase --- 
        # random rollout from this node to terminal state
        reward, simulation_path = self._simulate(expanded_node)
        self._log_simulation_result(simulation_path, reward)
        
        # --- Backpropagation phase ---
        # update node values starting with the expanded node
        expanded_node.backpropagate(reward)
        
        self.total_simulations_run += 1
    
    def _select(self, node: MCTSNode) -> (MCTSNode, List[MCTSNode]):
        """Select path through tree using UCT policy."""
        
        # initialize path
        path = [node]

        # so long as we are not at a leaf (i.e. node is fully expanded) 
        # and not terminal (max depth) get "best" child (UCT criteria)
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

        if node.is_terminal(self.max_depth):
            return node
        
        # Generate next message based on current turn
        current_state = node.state

        # go to the other agent
        next_turn = (current_state.current_turn + 1) % 2
        agent = next_turn + 1

        # get agent's message
        # prompt = self._build_prompt(current_state, agent=agent)
        next_message = self._get_agent_response(current_state, agent=agent)
        
        # create new state
        new_messages = current_state.messages + [next_message]
        new_state = ConversationState(
            messages=new_messages,
            current_turn=next_turn,
            depth=current_state.depth + 1
        )
        
        # Add child node
        child = node.add_child(next_message, new_state)
        self.total_nodes_created += 1

        return child
    
    def _simulate(self, node: MCTSNode) -> Tuple[float, List[str]]:
        """Simulate random conversation from current node to terminal state."""
        
        current_state = node.state
                
        # Continue simulation with random moves
        simulation_messages = current_state.messages.copy()
        simulation_turn = current_state.current_turn
        simulation_depth = current_state.depth
        
        while simulation_depth < self.rollout_depth:
            # Create temporary state for prompt building
            temp_state = ConversationState(
                messages=simulation_messages,
                current_turn=simulation_turn,
                depth=simulation_depth
            )

            # get next turn and agent
            simulation_turn = (simulation_turn + 1) % 2
            agent = simulation_turn + 1

            # get response
            # prompt = self._build_prompt(current_state, agent=agent)
            next_message = self._get_agent_response(current_state, agent=agent)
            
            simulation_messages.append(next_message)
            simulation_depth += 1
        
        # Calculate final reward
        final_state = ConversationState(
            messages=simulation_messages,
            current_turn=simulation_turn,
            depth=simulation_depth
        )
        
        reward = self.reward_function.calculate_reward(final_state)
        return reward, final_state.get_conversation_history()
    
    def _get_agent_response(self, state: ConversationState, agent: int) -> str:
        """Get response for Agent `agent` based on conversation history."""
        
        template = (
            "You are having a conversation with another agent. "
            "Here is the dialogue so far: "
            "{history}\n\n"
            "Continue this conversation as Agent {agent}. "
            "Keep your response limited to 2-3 sentences."
        )

        # Simple prompt: just the conversation history
        prompt = template.format(
            history=state.get_conversation_history(),
            agent=agent
        )

        response = self.agents[agent - 1].get_response(prompt)

        return response

    # ----------------------
    # Logging helper methods
    # ----------------------

    def _log_header(self, title: str):
        """Log a formatted header."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"{title:^80}")
        self.logger.info("=" * 80)
    
    def _log_separator(self):
        """Log a separator line."""
        self.logger.info("-" * 80)
    
    def _log(self, message: str):
        """Log an info message."""
        self.logger.info(message)
    
    def _log_candidate_header(self, candidate_num: int, total_candidates: int, candidate: str):
        """Log candidate evaluation header."""
        self.logger.info(f"\n{'='*20} CANDIDATE {candidate_num}/{total_candidates} {'='*20}")
        self.logger.info(f"Candidate Response: \"{candidate[:100]}{'...' if len(candidate) > 100 else ''}\"")
        self.logger.info("-" * 60)
    
    def _log_candidate_result(self, candidate_num: int, candidate: str, score: float):
        """Log candidate evaluation result."""
        self.logger.info(f"\nCandidate {candidate_num} Final Score: {score:.6f}")
        self.logger.info("=" * 60)
    
    def _log_simulation_header(self, simulation_num: int):
        """Log simulation header."""
        # if simulation_num <= 5 or simulation_num % 10 == 0:  # Log first 5 and every 10th
        self.logger.info(f"\n  --- Simulation {simulation_num} ---")
    
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

    def _log_simulation_result(self, simulation_path: List[str], reward: float):
        """Log simulation rollout and result."""
        # Only log detailed simulations for first few
        # if self.total_simulations_run < 5:
        if simulation_path:
            self.logger.info("  Simulation Rollout:")
            for step in simulation_path:
                # step_preview = self._response_preview(step)
                step_preview = step
                self.logger.info(f"    {step_preview}")
        self.logger.info(f"  Simulation Reward: {reward:.6f}")
        # elif self.total_simulations_run % 10 == 0:
        # self.logger.info(f"  Reward: {reward:.6f}")
    
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
                action_preview = child.action[:60] + "..." if child.action and len(child.action) > 60 else child.action
                self.logger.info(f"    {i+1}. \"{action_preview}\" -> visits={child.visits}, avg_reward={child.get_average_reward():.6f}")
    
    def _response_preview(self, response: str, max_length: int = 50) -> str:
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
