from reward_functions import Agent1LengthReward, Agent1WordCountReward, CompositeReward
from mcts_node import ConversationState

# Mock conversation states for testing
state1 = ConversationState(messages=[
    "Hello, how can I help you today?",  # Agent 1
    "What are the hours for the library?",  # Agent 2
    "The library is open from 8am to 6pm on weekdays.",  # Agent 1
    "Thank you!"  # Agent 2
], current_turn=0, depth=4)

state2 = ConversationState(messages=[
    "Hi!",  # Agent 1
    "Hello!",  # Agent 2
    "What's up?",  # Agent 1
    "Not much, you?"  # Agent 2
], current_turn=0, depth=4)

state3 = ConversationState(messages=[
    "This is a long, detailed answer from Agent 1 with many words included in the sentence.",
    "Short reply.",
    "Another lengthy and informative response from Agent 1, demonstrating verbosity.",
    "Concise."], current_turn=0, depth=4)

print("Testing Agent1LengthReward (raw):")
length_reward = Agent1LengthReward(normalize=False)
print(f"State 1: {length_reward.calculate_reward(state1)}")
print(f"State 2: {length_reward.calculate_reward(state2)}")
print(f"State 3: {length_reward.calculate_reward(state3)}")

print("\nTesting Agent1WordCountReward (raw):")
wordcount_reward = Agent1WordCountReward(normalize=False)
print(f"State 1: {wordcount_reward.calculate_reward(state1)}")
print(f"State 2: {wordcount_reward.calculate_reward(state2)}")
print(f"State 3: {wordcount_reward.calculate_reward(state3)}")

print("\nTesting CompositeReward (50% length, 50% word count):")
composite = CompositeReward([
    (Agent1LengthReward(normalize=False), 0.5),
    (Agent1WordCountReward(normalize=False), 0.5)
])
print(f"State 1: {composite.calculate_reward(state1)}")
print(f"State 2: {composite.calculate_reward(state2)}")
print(f"State 3: {composite.calculate_reward(state3)}")
