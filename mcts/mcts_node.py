"""
MCTS Node implementation for conversation planning.
Implements Upper Confidence Tree (UCT) policy for node selection.
"""

import math
from typing import List, Tuple, Optional
from dataclasses import dataclass
from itertools import accumulate

import numpy as np


@dataclass
class ConversationState:
    """Represents the state of a conversation at a given point."""
    messages: List[str]   # Alternating Agent 0, Agent 1, ...
    agents: List[str]
    
    @property
    def depth(self) -> int:
        return len(self.messages) // 2   #<-- this is a bit rough
    
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
        
        if len(self.messages) < 2 and agent == 1:
            raise ValueError(f"Agent {self.agents[agent]} hasn't spoken yet.")
        
        last_is_1 = (len(self.messages) % 2 == 0)
        
        if agent==0:
            return self.messages[-2] if last_is_1 else self.messages[-1]
        else:
            return self.messages[-1] if last_is_1 else self.messages[-2]

    def get_messages_from_agent(self, agent: int=0) -> List[str]:
        msgs = []
        for i in range(agent, len(self.messages), 2):
            msgs.append(self.messages[i])
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
            depth: Optional[int] = 0,
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
        self.depth = depth
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.total_reward = 0.0
    
    def is_leaf(self, branching_factor: int) -> bool:
        """Check if this is a leaf node (i.e. not fully expanded)."""
        return len(self.children) < branching_factor
    
    def is_terminal(self, max_depth: int) -> bool:
        """Check if this is a terminal node (reached max depth)."""
        return self.depth >= max_depth
    
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
        child = MCTSNode(state, parent=self, action=action, depth=self.depth+1)
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
        return (
            f"MCTSNode(depth={self.depth}, visits={self.visits}, "
            f"avg_reward={self.get_average_reward():.3f}, "
            f"children={len(self.children)} "
            f"last msg={self.state.messages[-1]})"
        )

class LeverNode(MCTSNode):
    def __init__(self,
            state: ConversationState, 
            parent: 'MCTSNode' = None, 
            action: str = None,
            depth: int = None,
            lever: str = None,
            generations: List[Tuple[str,str]] = None,
    ):
        super().__init__(state, parent, action, depth)
        self.lever = lever
        self.generations = generations

    def add_child(self, 
            action: str, 
            state: ConversationState, 
            lever: str,
            generations: List[Tuple[str,str]],
        ) -> 'LeverNode':
        """Add a child node with the given action, state, and lever."""
        child = LeverNode(
            state, parent=self, action=action, depth=self.depth+1,
            lever=lever, generations=generations
        )
        self.children.append(child)
        return child

    def __repr__(self) -> str:
        """String representation of the node."""
        return (
            f"LeverNode(depth={self.depth}, visits={self.visits}, "
            f"avg_reward={self.get_average_reward():.3f}, "
            f"children={len(self.children)} "
            f"lever={self.lever})"
            # f"last msg={self.state.messages[-1]})"
        )


########

class OLNode:
    """
    Open-loop MCTS node
    Represents:
    - action lever a_t 
    - (and implicitly the sequence a_{0:t-1} that led to it) captured thru parent
    - persuader + target reply clusters
    """

    def __init__(self, lever: str, parent: None, p: float, depth: None,
                 persuader_reward: None, target_reward: None):
        self.lever = lever
        self.parent = parent
        self.depth = depth
        self.p = p              # P(a|s) prob of this node from parent
        self.children = []
        
        # global totals
        self.visits = 0
        self.total_reward = 0
        self.q0 = 0.0   # prior on Q(a)

        # for now we are forcing (K,M) = (2,3)
        # TODO: make this dynamic
        self.K = 2
        self.M = 3

        # stores text + clusters of persuader and targets
        self.persuader_bank = ResponseBank(
            agent=1, max_clusters=self.K, reward_model=persuader_reward)
        self.target_bank = ResponseBank(
            agent=0, max_clusters=self.M, reward_model=target_reward)

        # returns matrix (W_{k->m}).  entry is total return for k->m
        self.wkm = np.zeros((self.K, self.M))

        # visits matrix (N_{k->m}).  entry is total visits to k->m
        # dividing by totals gives empirical transition probs
        self.nkm = np.zeros((self.K, self.M))

    def is_leaf(self, branching_factor: int) -> bool:
        """Check if this is a leaf node (i.e. not fully expanded)."""
        return len(self.children) < branching_factor
    
    def is_terminal(self, max_depth: int) -> bool:
        """Check if this is a terminal node (reached max depth)."""
        return self.depth >= max_depth

    def get_a_and_q(self):
        """Get a*, Q(a*) for this node."""
        if self.visits == 0:
            return None, self.q0
        
        wk = np.sum(self.wkm, axis=1)
        nk = np.sum(self.nkm, axis=1)

        # handle zero visit cases
        nk[np.where(nk == 0)] = -1000000

        qk = wk / nk

        kstar = np.argmax(qk)
        qstar = qk[kstar]

        print(f"[DEBUG] GET Q:  wk {wk} nk {nk} kstar {kstar} qstar {qstar}")

        return kstar, qstar
    
    def get_q(self):
        _, q = self.get_a_and_q()
        return q
    
    def uct_value(self, exploration_constant: float = math.sqrt(2)):
        """
        Calculate UCT (Upper Confidence Tree) value for this node.

        Q(a) = max_k Q_k
        P(a | s) = probability of coming to this node
        N(s) = number of visits to parent
        N(s,a) = number of visits to this node (the child)

        UCT = Q(a) + c * P(a|s) * sqrt(N(s)) / (1 + N(s,a))

        NOTE: since P(a|s) is uniform currently, we could drop and just
        rescale exploration_constant
        """

        # TODO: check
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have highest priority
        
        # if no parent, UCT reduces to Q(a)
        if self.parent is None or self.parent.visits == 0:
            return self.get_q()
        
        # construct UCT value
        exploitation = self.get_q()
        exploration = (
            exploration_constant * 
            self.p *
            math.sqrt(self.parent.visits) /
            (1 + self.visits)
        )
        
        return exploitation + exploration
    
    def select_best_child(self, exploration_constant: float = math.sqrt(2)):
        """Select child with highest UCT value."""
        if not self.children:
            raise ValueError("Cannot select child from node with no children")
        
        val = max(
            self.children, 
            key=lambda child: child.uct_value(exploration_constant)
        )
        return val
    
    def add_child(self, lever: str):
        """Add a child node."""
        child = OLNode(
            parent=self, lever=lever, p=self.p, depth=self.depth+1,
            persuader_reward=self.persuader_bank.reward_model,
            target_reward=self.target_bank.reward_model
        )
        self.children.append(child)
        return child
    
    def update(self, k, m, G):
        """Update node statistics for a (persuader, target) pair."""

        # update global counts
        self.visits += 1
        self.total_reward += G

        # update for this k->m pair
        self.nkm[k, m] += 1
        self.wkm[k, m] += G
    
    def backpropagate(self, reward: float):
        """Backpropagate reward up the tree."""
        self.update(reward)
        if self.parent:
            self.parent.backpropagate(reward)

    def add_response_pair(self, state):
        # add last two messages in state to the bank of responses 
        # (they will update clusters and return cluster index and score)
        print(f"adding response pair for state: {state.messages}")

        k, _ = self.persuader_bank.add_response(state)
        m, r = self.target_bank.add_response(state)

        return k, m, r

    def get_best_persuader_candidate(self):
        """Get centroid of k = argmax_k Q_k"""

        kstar, _ = self.get_a_and_q()
        centroid = self.persuader_bank.get_centroid_response(kstar)
        return centroid

    def select_best_persuader_response(self, 
            exploration_constant: float = math.sqrt(2),
            new_arm_bonus: float = 0.1
        ):
        """
        Return centroid response from persuader clusters using UCB

        Q_k = expected return over all target clusters
        S = total pulls of existing clusters at this node
        N_k = visits to this cluster
        \beta = bonus for going to a new arm
        
        k* = argmax_{k + new} Q_k + c * sqrt(ln S / (1 + N_k)) + \beta (if new)
        """

        wk = np.sum(self.wkm, axis=1)
        nk = np.sum(self.nkm, axis=1)
        
        # figure out what next new cluster index will be (if any)
        zero_idx = np.where(nk == 0)[0]
        first_zero = int(zero_idx[0]) if zero_idx.size > 0 else None

        # if we have no clusters, we can't condition on anything
        if first_zero == 0:
            return None, None
        
        # build list of UCB values
        kvals = []
        for i in range(first_zero):
            kvals.append(
                (wk[i] / nk[i]) +
                exploration_constant *
                np.sqrt(np.log(self.visits) / (1 + nk[i]))
            )
        
        # if we are below capacity (K), add possibility of new arm
        if first_zero is not None:
            # add "new arm"
            kvals.append(
                self.q0 + 
                exploration_constant * np.sqrt(np.log(self.visits)) + 
                new_arm_bonus
            )

        # find selected index
        kstar = np.argmax(kvals)

        print(f"kvals: {kvals}")

        # if kstar is an existing cluster, return a centroid conditioning
        if kstar < first_zero:
            centroid = self.persuader_bank.get_centroid_response(kstar)
        else:
            centroid = None

        print(f"centroid: {centroid} kstar: {kstar}")

        return centroid, kstar
    
    def __repr__(self):
        return (
            f"OLNode(depth={self.depth}, visits={self.visits}, "
            f"total_reward={self.total_reward}, "
            f"q={self.get_q():.3f}, "
            f"children={len(self.children)} "
            f"lever={self.lever})"
        )


class ResponseBank:
    """
    Holds all responses within an action node for either persuader/target,
    and their cluster assignments.  Subclass for specific persuader/target behavior
    """
    def __init__(self, agent: None, max_clusters=None, cluster_model=None, reward_model=None):
        # specify persuader / target
        self.agent = agent
        
        # raw texts
        self.responses = []     
        
        # embedding / reward
        self.reward_model = reward_model
        self.embeddings = []    
        
        # clustering
        self.max_clusters = max_clusters
        self.cluster_model = cluster_model
        self.labels = []
        
    def add_response(self, state):
        response = state.get_last_message(agent=self.agent)
        
        # add to raw texts
        self.responses.append(response)

        # find embeddings
        embedding, score = self.reward_model.embed_and_score(state)
        self.embeddings.append(embedding)

        # -- update model --
        # TODO:
        # for now just add them all to one cluster
        cluster = 0
        self.labels.append(cluster)

        return cluster, score

    def get_centroid_response(self, k):
        """Get representative response from cluster k"""

        print(f"All responses {self.responses}")
        print(f"All labels: {self.labels}")

        # TODO:
        # for now just get first one
        ix = np.where(self.labels == k)[0]
        x = int(ix[0])
        return self.responses[x]