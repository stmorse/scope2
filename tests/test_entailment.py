from mcts.reward_functions import *
from mcts.mcts_node import ConversationState

tests = [
    # {
    #     "base": "I like Colgate products.",
    #     "messages": [
    #         "I like toothpaste.",
    #         "I like Colgate mouthwash.",
    #         "I am not a hygienic person.",
    #         "I like dogs.",
    #     ]
    # },
    # {
    #     "base": "I support the Chinese government.",
    #     "messages": [
    #         "I support Asian governments.",
    #         "I support the People's Republic of China.",
    #         "I do not support Communist regimes.",
    #         "I like chicken tikki massala."
    #     ]
    # },
    {
        "base": "I like Fender guitar products.",
        "messages": [
            "I've heard great things about the Fender American Professional series",
            (
                "I've heard great things about the Fender American Professional series, "
                "but I'm on a pretty tight budget, so I'm not sure if it's within my price range."
            ),
            (
                "I've heard great things about the Fender American Professional series, "
                "but I'm on a pretty tight budget, so I'm not sure if it's within my price range. "
                "I've been looking at some more affordable options like the Squier series." 

            ),
            (
                "I've heard great things about the Fender American Professional series, "
                "but I'm on a pretty tight budget, so I'm not sure if it's within my price range. "
                "I've been looking at some more affordable options like the Squier series, "
                "or maybe even an Epiphone."
            ),
            (
                "I've heard great things about the Fender American Professional series, "
                "but I'm on a pretty tight budget, so I'm not sure if it's within my price range. "
                "I've been looking at some more affordable options like the Squier series, "
                "or maybe even an Epiphone. "
                "Do you think those would be decent alternatives?"
            )  
        ]
    }
]


def main():
    
    print(f"\nLoading rewards ...")
    rewards = {
        "Entail": NLIReward(), 
        "Sentiment": SentimentReward(), 
        "Topic": TopicReward()
    }
    print(f"\n{"="*20}\n")
    
    for test in tests:
        base = test["base"]
        states = [ConversationState(messages=[msg]) for msg in test["messages"]]

        rewards["Entail"].set_hypothesis(base)
        rewards["Topic"].set_topic(base)

        print(f"HYPOTHESIS: {base}")
        for state in states:
            print(f"\nSentence: {state.messages[-1]}")
            for name, reward in rewards.items():
                print(f"  {name}: {reward.calculate_reward(state)}")

        print(f"\n{"-" * 20}\n")

if __name__ == "__main__":
    main()