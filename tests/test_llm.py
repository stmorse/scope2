from agent.agent import Agent
from mcts.mcts_node import ConversationState


state = ConversationState(messages=["What are your thoughts on AI?"])

agent = Agent(
    name="Agent 1", 
    provider="ollama", 
    model="llama3.2:latest",
    forcing=False
)

# top_p
# if top_p=0.9, the algorithm keeps the smallest number of top tokens 
# whose combined probability is at least 90%.

# top_k -- keeps top k most probable tokens

# for temp in [5, 100]:
#     print(f"\n{"="*20}\nTEMPERATURE: {temp:.1f}\n")
#     for _ in range(5):
#         response = agent.get_response(
#             state, temperature=temp, top_p=0.99, top_k=10000, min_p=0.001)
#         print(response)

print(state)
response = agent.get_response(state)
print(response)

print()