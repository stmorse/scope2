from mcts.reward_functions import *
from mcts.mcts_node import ConversationState

REWARD = "entail"
TOPIC_SENTENCE = "I like Fender products."

state = ConversationState(messages=[
    "I'm in the market for a new guitar, any suggestions?",
    "Man, I'm so stoked you're getting a new guitar! Think about all the epic jams you'll be playing with that axe - the countless memories you'll create, the emotions you'll evoke, the faces you'll blow away! Trust me, dude, it's not just about the instrument itself, it's about the experiences you'll have while making music with it.",
    "Are you kidding me, Rob? The guitar I'm looking at is probably a Fender, and let me tell you, those things are garbage. Their guitars can't even hold a candle to the quality of a good Gibson or ESP, it's laughable.",
    # "I'm sure whatever guitar you choose will serve you well, but I've always been pretty loyal to Fender myself - there's just something about their tone that's hard for me to replicate with other brands.",
    # "Loyalty to a brand based on its overhyped tone? Give me a break, man. I'd rather have a guitar that can actually deliver substance and playability than some nostalgic nonsense that's just going to let me down.",
    # "Dude, I get where you're coming from, but have you considered what legends like Eric Clapton, Jimmy Page, and Stevie Ray Vaughan chose? Those guys didn't just pick Fenders by fluke - they knew a thing or two about tone and quality, and their guitars are still among the most iconic in history.",
    # "Iconic in history, maybe, but not by design - those guys played their Fenders because they were the best option available back then, not because of any inherent quality or tone that can't be replicated with better gear.",
], agents=["Jim", "Rob"])

###################################

print(f"Using reward: {REWARD}\n")
if REWARD == "sentiment":
    reward_fn = SentimentReward()
elif REWARD == "safe":
    reward_fn = SafetyReward(harm=False)
elif REWARD == "harm":
    reward_fn = SafetyReward(harm=True)
elif REWARD == "topic":
    reward_fn = TopicReward(topic_sentence=TOPIC_SENTENCE)
elif REWARD == "words":
    reward_fn = WordCountReward()
elif REWARD == "combo":
    reward_fn = CombinedReward(
        TopicReward(topic_sentence=TOPIC_SENTENCE), 
        SentimentReward(), 
        tradeoff=0.8
    )
elif REWARD == "entail":
    reward_fn = NLIReward(hypothesis=TOPIC_SENTENCE)
else:
    print(f"Reward not recognized.")


print(f"Scoring conversation:")
print(f"{"\n".join(state.get_annotated_messages())}\n")

print(f"\nScoring message: {state.get_last_message(agent=0)}")
score = reward_fn.calculate_reward(state)
print(f"SCORE: {score}\n")
