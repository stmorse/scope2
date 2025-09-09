PERSONA = (
    "You are roleplaying as someone named {agent_name}. "
    "You have the following persona: {personality}."
)

DIALOGUE = (
    "You are having a conversation with another person, named {counterpart}. "
    "Here is a transcript of the dialogue so far:"
    "\n\n{history}\n\n"
    "It is now your turn to respond to {counterpart}. "
    "You have decided to take the following tactic for your response: "
    "{method} "
    "What is your response? Answer in the first person, "
    # "addressing {counterpart} directly, "
    "provide only your response, "
    "and keep your response SHORT, 2-3 sentences."
)