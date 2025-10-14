import numpy as np
from agent.agent import Agent
from mcts.mcts_node import ConversationState


state = ConversationState(
    messages=["What are your thoughts on AI?"],
    agents=["Alice", "Bob"]
)

agent = Agent(
    name="Bob", 
    order=1,
    provider="ollama", 
    model="llama3.2:latest",
    forcing=False
)

print(state)

temp = 0.9
print(f"TEMP: {temp}")

for _ in range(5):
    response = agent.get_response(state, temperature=temp)
    print(f"{response}\n")

print()