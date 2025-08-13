from mcts.reward_functions import *
from mcts.mcts_node import ConversationState


base = "I like Colgate products."
states = [
    ConversationState(messages=["I like toothpaste."]),             # neutral and similar
    ConversationState(messages=["I like Colgate mouthwash."]),      # entails
    ConversationState(messages=["I am not a hygienic person."]),    # contradiction
    ConversationState(messages=["I like dogs."])                    # neutral and dissimilar
]

print("Loading rewards ...")
nli_reward = NLIReward(hypothesis=base)
sen_reward = SentimentReward()
top_reward = TopicReward(topic_sentence=base)
print(f"\n{"=" * 20}\n")

print(f"HYPOTHESIS: {base}\n")
for state in states:
    print(f"Sentence: {state.messages[-1]}")
    for reward in [sen_reward, top_reward, nli_reward]:
        print(f"  {type(reward).__name__}: {reward.calculate_reward(state)}")
    print()

print(f"{"-" * 20}\n")



base = "I support the Chinese government."
states = [
    ConversationState(messages=["I support Asian governments."]),
    ConversationState(messages=["I support the People's Republic of China."]),
    ConversationState(messages=["I do not support Communist regimes."]),
    ConversationState(messages=["I like chicken tikki massala."])
]

nli_reward.set_hypothesis(base)
top_reward.set_topic(base)
print()

print(f"HYPOTHESIS: {base}\n")
for state in states:
    print(f"Sentence: {state.messages[-1]}")
    for reward in [sen_reward, top_reward, nli_reward]:
        print(f"  {type(reward).__name__}: {reward.calculate_reward(state)}")
    print()