"""
Main MCTS Conversation Planner implementation.
Uses Monte Carlo Tree Search with UCT policy to select optimal LLM responses.
"""

import random
import math
from typing import List, Tuple, Optional
from llm_providers import LLMProvider
from mcts_node import MCTSNode, ConversationState
from reward_functions import RewardFunction, DEFAULT_REWARD_FUNCTION


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
                 temperature: float = 0.7):
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
        """
        self.agent1_provider = agent1_provider
        self.agent2_provider = agent2_provider
        self.reward_function = reward_function or DEFAULT_REWARD_FUNCTION
        self.max_depth = max_depth
        self.num_simulations = num_simulations
        self.exploration_constant = exploration_constant
        self.temperature = temperature
    
    def plan_conversation(self, 
                         initial_prompt: str, 
                         num_candidates: int = 5) -> List[Tuple[str, float]]:
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
        
        # Evaluate each candidate using MCTS
        results = []
        for i, candidate in enumerate(candidates):
            print(f"Evaluating candidate {i+1}/{len(candidates)}")
            score = self._evaluate_candidate(initial_prompt, candidate)
            results.append((candidate, score))
        
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
        for _ in range(self.num_simulations):
            self._run_simulation(root)
        
        return root.get_average_reward()
    
    def _run_simulation(self, root: MCTSNode):
        """Run a single MCTS simulation from the root node."""
        # Selection: traverse tree using UCT until leaf
        node = self._select(root)
        
        # Expansion: add new child if not terminal
        if not node.is_terminal(self.max_depth):
            node = self._expand(node)
        
        # Simulation: random rollout from current node
        reward = self._simulate(node)
        
        # Backpropagation: update node values
        node.backpropagate(reward)
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select path through tree using UCT policy."""
        while not node.is_leaf() and not node.is_terminal(self.max_depth):
            node = node.select_best_child(self.exploration_constant)
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
        return child
    
    def _simulate(self, node: MCTSNode) -> float:
        """Simulate random conversation from current node to terminal state."""
        current_state = node.state
        
        # If already terminal, calculate reward
        if current_state.depth >= self.max_depth:
            return self.reward_function.calculate_reward(current_state)
        
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
                simulation_turn = 1
            else:
                # Agent 2's turn
                prompt = self._build_agent2_prompt(temp_state)
                next_message = self.agent2_provider.generate_response(prompt, self.temperature)
                simulation_turn = 0
            
            if not next_message:
                break  # If generation fails, end simulation
            
            simulation_messages.append(next_message)
            simulation_depth += 1
        
        # Calculate final reward
        final_state = ConversationState(
            messages=simulation_messages,
            current_turn=simulation_turn,
            depth=simulation_depth
        )
        
        return self.reward_function.calculate_reward(final_state)
    
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
            "agent2_provider": type(self.agent2_provider).__name__
        }
