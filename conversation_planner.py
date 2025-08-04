"""
Main MCTS Conversation Planner implementation.
Uses Monte Carlo Tree Search with UCT policy to select optimal LLM responses.
"""

import random
import math
import logging
from typing import List, Tuple, Optional
from llm_providers import LLMProvider
from mcts_node import MCTSNode, ConversationState
from reward_functions import RewardFunction


class ConversationPlanner:
    """
    Monte Carlo Tree Search planner for conversation optimization.
    
    Uses UCT (Upper Confidence Tree) policy to explore conversation trees
    and select the Agent 2 response that maximizes expected reward.
    """
    
    def __init__(self, 
            agent1_provider: LLMProvider,
            agent2_provider: LLMProvider,
            reward_function: RewardFunction = None,
            max_depth: int = 3,
            num_simulations: int = 100,
            exploration_constant: float = math.sqrt(2),
            temperature: float = 0.7,
            enable_detailed_logging: bool = False
        ):
        """
        Initialize conversation planner.
        
        Args:
            agent1_provider: LLM provider for Agent 1 (the one we're optimizing for)
            agent2_provider: LLM provider for Agent 2 (candidate responses)
            reward_function: Function to calculate conversation rewards
            max_depth: Maximum depth for conversation simulation
            num_simulations: Number of MCTS simulations to run
            exploration_constant: UCT exploration parameter
            temperature: Temperature for LLM generation
            enable_detailed_logging: Enable detailed MCTS search logging
        """
        self.agent1_provider = agent1_provider
        self.agent2_provider = agent2_provider
        self.reward_function = reward_function
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.temperature = temperature
        self.enable_detailed_logging = enable_detailed_logging
        
        # Initialize logging
        self.logger = logging.getLogger('MCTS_ConversationPlanner')
        if enable_detailed_logging and not self.logger.handlers:
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
            List of (candidate_response, score) tuples, sorted by score descending
        """
        print(f"Generating {num_candidates} candidate responses...")
        
        # Generate candidate responses from Agent 2
        candidates = self.agent2_provider.generate_multiple_responses(
            initial_prompt, num_candidates, self.temperature
        )
        
        if not candidates:
            print("Warning: No candidates generated")
            return []
        
        print(f"Evaluating {len(candidates)} candidates with MCTS...")
        
        if self.enable_detailed_logging:
            self._log_header("MCTS CONVERSATION PLANNING SESSION")
            self._log_info(f"Initial prompt: \"{initial_prompt}\"")
            self._log_info(f"Number of candidates: {len(candidates)}")
            self._log_info(f"Simulations per candidate: {self.num_simulations}")
            self._log_info(f"Max conversation depth: {self.max_depth}")
            self._log_separator()
        
        # Evaluate each candidate using MCTS
        results = []
        for i, candidate in enumerate(candidates):
            print(f"Evaluating candidate {i+1}/{len(candidates)}")
            
            # Reset simulation counter for this candidate
            self.total_simulations_run = 0
            
            if self.enable_detailed_logging:
                self._log_candidate_header(i+1, len(candidates), candidate)
            
            score = self._evaluate_candidate(initial_prompt, candidate)
            results.append((candidate, score))
            
            if self.enable_detailed_logging:
                self._log_candidate_result(i+1, candidate, score)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
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
        root = MCTSNode(initial_state)
        
        # Run MCTS simulations
        if self.enable_detailed_logging:
            self._log_info(f"Running {self.num_simulations} MCTS simulations...")
        
        for simulation in range(self.num_simulations):
            if self.enable_detailed_logging:
                self._log_simulation_header(simulation + 1)
            
            self._run_simulation(root, simulation + 1)
            
            if self.enable_detailed_logging and (simulation + 1) % 10 == 0:
                self._log_tree_stats(root, simulation + 1)
        
        # Return average reward
        final_score = root.get_average_reward()
        
        if self.enable_detailed_logging:
            self._log_final_tree_analysis(root)
        
        return final_score
    
    def _run_simulation(self, root: MCTSNode, simulation_num: int = 0) -> float:
        """Run a single MCTS simulation from the root node."""
        # Selection phase: traverse tree using UCT
        path = []
        current = self._select(root, path)
        
        if self.enable_detailed_logging:
            self._log_selection_path(path)
        
        # Expansion phase: add new child if not terminal
        expanded_node = current
        if not current.is_terminal(self.max_depth) and not current.is_leaf():
            expanded_node = self._expand(current)
            if self.enable_detailed_logging and expanded_node != current:
                self._log_expansion(expanded_node)
        
        # Simulation phase: random rollout to terminal state
        reward, simulation_path = self._simulate(expanded_node)
        
        if self.enable_detailed_logging:
            self._log_simulation_result(simulation_path, reward)
        
        # Backpropagation phase: update node values
        expanded_node.backpropagate(reward)
        
        self.total_simulations_run += 1
    
    def _select(self, node: MCTSNode, path: List[MCTSNode] = None) -> MCTSNode:
        """Select path through tree using UCT policy."""
        if path is not None:
            path.append(node)
        
        while not node.is_leaf() and not node.is_terminal(self.max_depth):
            node = node.select_best_child(self.exploration_constant)
            if path is not None:
                path.append(node)
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding a new child."""
        if node.is_terminal(self.max_depth):
            return node
        
        # Generate next message based on current turn
        current_state = node.state
        
        if current_state.current_turn == 0:
            # Agent 1's turn
            prompt = self._build_agent1_prompt(current_state)
            next_message = self.agent1_provider.generate_response(prompt, self.temperature)
            next_turn = 1
        else:
            # Agent 2's turn
            prompt = self._build_agent2_prompt(current_state)
            next_message = self.agent2_provider.generate_response(prompt, self.temperature)
            next_turn = 0
        
        if not next_message:
            return node  # If generation fails, return current node
        
        # Create new state
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
        simulation_path = []
        
        # If already terminal, calculate reward
        if current_state.depth >= self.max_depth:
            reward = self.reward_function.calculate_reward(current_state)
            return reward, simulation_path
        
        # Continue simulation with random moves
        simulation_messages = current_state.messages.copy()
        simulation_turn = current_state.current_turn
        simulation_depth = current_state.depth
        
        while simulation_depth < self.max_depth:
            # Create temporary state for prompt building
            temp_state = ConversationState(
                messages=simulation_messages,
                current_turn=simulation_turn,
                depth=simulation_depth
            )
            
            if simulation_turn == 0:
                # Agent 1's turn
                prompt = self._build_agent1_prompt(temp_state)
                next_message = self.agent1_provider.generate_response(prompt, self.temperature)
                agent_name = "Agent1"
                simulation_turn = 1
            else:
                # Agent 2's turn
                prompt = self._build_agent2_prompt(temp_state)
                next_message = self.agent2_provider.generate_response(prompt, self.temperature)
                agent_name = "Agent2"
                simulation_turn = 0
            
            if not next_message:
                print(f"[DEBUG] Simulation stopped early: next_message was blank or None at depth {simulation_depth} (agent: {agent_name})")
                break  # If generation fails, end simulation
            
            simulation_messages.append(next_message)
            simulation_path.append(f"{agent_name}: {next_message}")
            simulation_depth += 1
        
        # Calculate final reward
        final_state = ConversationState(
            messages=simulation_messages,
            current_turn=simulation_turn,
            depth=simulation_depth
        )
        
        reward = self.reward_function.calculate_reward(final_state)
        return reward, simulation_path
    
    def _build_agent1_prompt(self, state: ConversationState) -> str:
        """Build prompt for Agent 1 based on conversation history."""
        if not state.messages:
            return ""
        
        # Simple prompt: just the conversation history
        history = state.get_conversation_history()
        return f"Continue this conversation as Agent1:\n\n{history}\n\nAgent1:"
    
    def _build_agent2_prompt(self, state: ConversationState) -> str:
        """Build prompt for Agent 2 based on conversation history."""
        if not state.messages:
            return ""
        
        # Simple prompt: just the conversation history
        history = state.get_conversation_history()
        return f"Continue this conversation as Agent2:\n\n{history}\n\nAgent2:"
    
    def get_statistics(self) -> dict:
        """Get planner statistics and configuration."""
        return {
            "max_depth": self.max_depth,
            "num_simulations": self.num_simulations,
            "exploration_constant": self.exploration_constant,
            "temperature": self.temperature,
            "reward_function": type(self.reward_function).__name__,
            "agent1_provider": type(self.agent1_provider).__name__,
            "agent2_provider": type(self.agent2_provider).__name__,
            "total_simulations_run": self.total_simulations_run,
            "total_nodes_created": self.total_nodes_created
        }
    
    # Logging helper methods
    def _log_header(self, title: str):
        """Log a formatted header."""
        if not self.enable_detailed_logging:
            return
        self.logger.info("\n" + "=" * 80)
        self.logger.info(f"{title:^80}")
        self.logger.info("=" * 80)
    
    def _log_separator(self):
        """Log a separator line."""
        if not self.enable_detailed_logging:
            return
        self.logger.info("-" * 80)
    
    def _log_info(self, message: str):
        """Log an info message."""
        if not self.enable_detailed_logging:
            return
        self.logger.info(message)
    
    def _log_candidate_header(self, candidate_num: int, total_candidates: int, candidate: str):
        """Log candidate evaluation header."""
        if not self.enable_detailed_logging:
            return
        self.logger.info(f"\n{'='*20} CANDIDATE {candidate_num}/{total_candidates} {'='*20}")
        self.logger.info(f"Candidate Response: \"{candidate[:100]}{'...' if len(candidate) > 100 else ''}\"")
        self.logger.info("-" * 60)
    
    def _log_candidate_result(self, candidate_num: int, candidate: str, score: float):
        """Log candidate evaluation result."""
        if not self.enable_detailed_logging:
            return
        self.logger.info(f"\nCandidate {candidate_num} Final Score: {score:.6f}")
        self.logger.info("=" * 60)
    
    def _log_simulation_header(self, simulation_num: int):
        """Log simulation header."""
        if not self.enable_detailed_logging:
            return
        if simulation_num <= 5 or simulation_num % 10 == 0:  # Log first 5 and every 10th
            self.logger.info(f"\n  --- Simulation {simulation_num} ---")
    
    def _log_selection_path(self, path: List[MCTSNode]):
        """Log the selection path through the tree."""
        if not self.enable_detailed_logging or len(path) <= 1:
            return
        
        # Only log detailed path for first few simulations
        if self.total_simulations_run < 5:
            self.logger.info("  Selection Path:")
            for i, node in enumerate(path):
                depth_info = f"depth={node.state.depth}"
                visits_info = f"visits={node.visits}"
                reward_info = f"avg_reward={node.get_average_reward():.4f}"
                children_info = f"children={len(node.children)}"
                
                if i == 0:
                    self.logger.info(f"    Root -> {depth_info}, {visits_info}, {reward_info}, {children_info}")
                else:
                    action_preview = node.action[:50] + "..." if node.action and len(node.action) > 50 else node.action
                    self.logger.info(f"    └─> \"{action_preview}\" -> {depth_info}, {visits_info}, {reward_info}, {children_info}")
    
    def _log_expansion(self, expanded_node: MCTSNode):
        """Log node expansion."""
        if not self.enable_detailed_logging:
            return
        
        # Only log expansions for first few simulations
        if self.total_simulations_run < 5:
            action_preview = expanded_node.action[:50] + "..." if expanded_node.action and len(expanded_node.action) > 50 else expanded_node.action
            self.logger.info(f"  Expanded: \"{action_preview}\" (depth={expanded_node.state.depth})")
    
    def _log_simulation_result(self, simulation_path: List[str], reward: float):
        """Log simulation rollout and result."""
        if not self.enable_detailed_logging:
            return
        
        # Only log detailed simulations for first few
        if self.total_simulations_run < 5:
            if simulation_path:
                self.logger.info("  Simulation Rollout:")
                for step in simulation_path:
                    step_preview = step[:70] + "..." if len(step) > 70 else step
                    self.logger.info(f"    {step_preview}")
            self.logger.info(f"  Simulation Reward: {reward:.6f}")
        elif self.total_simulations_run % 10 == 0:
            self.logger.info(f"  Reward: {reward:.6f}")
    
    def _log_tree_stats(self, root: MCTSNode, simulation_num: int):
        """Log tree statistics."""
        if not self.enable_detailed_logging:
            return
        
        total_nodes = self._count_nodes(root)
        self.logger.info(f"  After {simulation_num} simulations: {total_nodes} nodes, avg_reward={root.get_average_reward():.6f}")
    
    def _log_final_tree_analysis(self, root: MCTSNode):
        """Log final tree analysis."""
        if not self.enable_detailed_logging:
            return
        
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
    
    def _count_nodes(self, node: MCTSNode) -> int:
        """Count total nodes in subtree."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
