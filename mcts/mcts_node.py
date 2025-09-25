"""
MCTS Node implementation for conversation planning.
Implements Upper Confidence Tree (UCT) policy for node selection.
"""

import math
from typing import List, Optional
from dataclasses import dataclass
from itertools import accumulate


@dataclass
class ConversationState:
    """Represents the state of a conversation at a given point."""
    messages: List[str]   # Alternating Agent 0, Agent 1, ...
    agents: List[str]
    
    @property
    def depth(self) -> int:
        return len(self.messages)
    
    @property
    def current_turn(self) -> int:
        """Which agent we are awaiting response from"""
        return self.depth % 2   # 1 (A0 just spoke) -> 1 (awaiting A1), 2 -> 0

    def add_message(self, message: str) -> 'ConversationState':
        new_messages = self.messages.copy()
        new_messages.append(message)
        return ConversationState(messages=new_messages, agents=self.agents)

    def get_last_message(self, agent: int=0) -> str:
        # agent=0 --> depth even, go back two, depth odd, get last
        # agent=1 --> depth even, get last, depth odd, go back two
        
        if self.depth < 2 and agent == 1:
            raise ValueError(f"Agent {self.agents[agent]} hasn't spoken yet.")
        
        last_is_1 = (self.depth % 2 == 0)
        
        if agent==0:
            return self.messages[-2] if last_is_1 else self.messages[-1]
        else:
            return self.messages[-1] if last_is_1 else self.messages[-2]

    def get_messages_from_agent(self, agent: int=0) -> List[str]:
        msgs = []
        for i, msg in enumerate(self.messages):
            if i % 2 == agent:
                msgs.append(msg)
        return msgs
    
    def get_annotated_messages(self) -> str:
        """Get formatted conversation history."""
        history = []
        for i, message in enumerate(self.messages):
            agent = self.agents[i % 2]
            history.append(f"{agent}: {message}")
        return history
    
    def get_annotated_messages2(self, whoami: int) -> str:
        history = []
        for i, message in enumerate(self.messages):
            anum = (i % 2)
            agent = "Me" if whoami == anum else self.agents[anum]
            history.append(f"{agent}: {message}")
        return history

    def convert_to_chat(self) -> List[dict]:
        """
        Convert to API-style chat history for some embedders  
        Assumes Agent 0=user, Agent 1=assistant
        """
        chat = []
        for i, message in enumerate(self.messages):
            role = "user" if i % 2 == 0 else "assistant"
            chat.append({"role": role, "content": message})
        return chat

    def convert_to_cumulative(self) -> List[str]:
        """Convert to cumulative list of strings"""
        cumulative = list(accumulate(self.messages, lambda x, y: f"{x}\n{y}"))
        return cumulative
    
    def get_deep_copy(self):
        new_state = ConversationState(
            messages=self.messages.copy(),
            agents=self.agents.copy()
        )
        return new_state

class MCTSNode:
    """Node in the MCTS tree for conversation planning."""
    
    def __init__(self, 
            state: ConversationState, 
            parent: Optional['MCTSNode'] = None, 
            action: Optional[str] = None,
        ):
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
    
    def is_leaf(self, branching_factor: int) -> bool:
        """Check if this is a leaf node (i.e. not fully expanded)."""
        return len(self.children) < branching_factor
    
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
        
        val = max(
            self.children, 
            key=lambda child: child.uct_value(exploration_constant)
        )
        return val
    
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
                f"children={len(self.children)} "
                f"last msg={self.state.messages[-1]})"
                )

class LeverNode(MCTSNode):
    def __init__(self,
            state: ConversationState, 
            parent: 'MCTSNode' = None, 
            action: str = None,
            lever: str = None,
    ):
        super().__init__(state, parent, action)
        self.lever = lever

    def add_child(self, action: str, state: ConversationState, lever: str) -> 'LeverNode':
        """Add a child node with the given action, state, and lever."""
        child = LeverNode(state, parent=self, action=action, lever=lever)
        self.children.append(child)
        return child

    def __repr__(self) -> str:
        """String representation of the node."""
        return (
            f"LeverNode(depth={self.state.depth}, visits={self.visits}, "
            f"avg_reward={self.get_average_reward():.3f}, "
            f"children={len(self.children)} "
            f"lever={self.lever})"
            # f"last msg={self.state.messages[-1]})"
        )
