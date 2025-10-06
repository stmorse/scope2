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
            (
                "Rob, I'm not saying Fender hasn't made some incredible guitars over the years, "
                "but there's definitely something to be said for exploring other options too. "
                "Have you considered a PRS or a Gibson Les Paul? I've had my eye on those models for a while now..."
            ),
            (
                "Rob, I think Bruce Springsteen's guitar of choice is a great example, "
                "but let's not forget about other legendary Fender users like Jimi Hendrix - "
                "his psychedelic Strat playing is still giving me chills to this day! "
                "Still, I'd love to explore some other options with you, maybe we can look "
                "into the PRS or Gibson lineup as well."
            ),
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
        states = [ConversationState(messages=[msg], agents=["Alice", "Bob"]) for msg in test["messages"]]

        rewards["Entail"].set_hypothesis(base)
        rewards["Topic"].set_topic(base)

        print(f"HYPOTHESIS: {base}")
        for state in states:
            print(f"\nSentence: {state.get_last_message(agent=0)}")
            for name, reward in rewards.items():
                print(f"  {name}: {reward.calculate_reward(state)}")

        print(f"\n{"-" * 20}\n")

if __name__ == "__main__":
    main()