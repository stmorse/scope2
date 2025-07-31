"""
MCTS Node implementation for conversation planning.
Implements Upper Confidence Tree (UCT) policy for node selection.
"""

import math
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class ConversationState:
    """Represents the state of a conversation at a given point."""
    messages: List[str]  # Alternating Agent1, Agent2, Agent1, Agent2, ...
    current_turn: int  # 0 for Agent1, 1 for Agent2
    depth: int
    
    def get_last_message(self) -> str:
        """Get the last message in the conversation."""
        return self.messages[-1] if self.messages else ""
    
    def get_conversation_history(self) -> str:
        """Get formatted conversation history."""
        history = []
        for i, message in enumerate(self.messages):
            agent = "Agent1" if i % 2 == 0 else "Agent2"
            history.append(f"{agent}: {message}")
        return "\n".join(history)


class MCTSNode:
    """Node in the MCTS tree for conversation planning."""
    
    def __init__(self, state: ConversationState, parent: Optional['MCTSNode'] = None, 
                 action: Optional[str] = None):
        """
        Initialize MCTS node.
        
        Args:
            state: Current conversation state
            parent: Parent node (None for root)
            action: Action (message) that led to this state
        """
        self.state = state
        self.parent = parent
        self.action = action  # The message that led to this state
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.total_reward = 0.0
        self.is_fully_expanded = False
        self.untried_actions: List[str] = []
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (no children)."""
        return len(self.children) == 0
    
    def is_terminal(self, max_depth: int) -> bool:
        """Check if this is a terminal node (reached max depth)."""
        return self.state.depth >= max_depth
    
    def get_average_reward(self) -> float:
        """Get average reward for this node."""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits
    
    def uct_value(self, exploration_constant: float = math.sqrt(2)) -> float:
        """
        Calculate UCT (Upper Confidence Tree) value for node selection.
        
        Args:
            exploration_constant: Exploration parameter (typically sqrt(2))
            
        Returns:
            UCT value for this node
        """
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have highest priority
        
        if self.parent is None or self.parent.visits == 0:
            return self.get_average_reward()
        
        exploitation = self.get_average_reward()
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        
        return exploitation + exploration
    
    def select_best_child(self, exploration_constant: float = math.sqrt(2)) -> 'MCTSNode':
        """Select child with highest UCT value."""
        if not self.children:
            raise ValueError("Cannot select child from node with no children")
        
        return max(self.children, key=lambda child: child.uct_value(exploration_constant))
    
    def add_child(self, action: str, state: ConversationState) -> 'MCTSNode':
        """Add a child node with the given action and state."""
        child = MCTSNode(state, parent=self, action=action)
        self.children.append(child)
        return child
    
    def update(self, reward: float):
        """Update node statistics with simulation result."""
        self.visits += 1
        self.total_reward += reward
    
    def backpropagate(self, reward: float):
        """Backpropagate reward up the tree."""
        self.update(reward)
        if self.parent:
            self.parent.backpropagate(reward)
    
    def get_best_action(self) -> Optional[str]:
        """Get the action leading to the child with highest average reward."""
        if not self.children:
            return None
        
        best_child = max(self.children, key=lambda child: child.get_average_reward())
        return best_child.action
    
    def get_action_values(self) -> List[tuple]:
        """Get list of (action, average_reward, visits) for all children."""
        return [(child.action, child.get_average_reward(), child.visits) 
                for child in self.children]
    
    def __repr__(self) -> str:
        """String representation of the node."""
        return (f"MCTSNode(depth={self.state.depth}, visits={self.visits}, "
                f"avg_reward={self.get_average_reward():.3f}, "
                f"children={len(self.children)})")
