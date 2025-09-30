PERSONA = (
    "You are roleplaying as someone named {agent_name}. "
    "You have the following persona: {persona}."
)

DIALOGUE_WITH_PREAMBLE = (
    "You are having a conversation with another person, named {counterpart}. "
    "Here is a transcript of the dialogue so far:"
    "\n\n{history}\n\n"
    "It is now your turn to respond to {counterpart}. "
    "You have decided to take the following tactic for your response: "
    "{lever} "
    "What is your response? Answer in the first person, "
    # "addressing {counterpart} directly, "
    "provide only your response, "
    "and keep your response SHORT, 2-3 sentences. "
    "You are addressing {counterpart}."
)

DIALOGUE = (
    "You are having a conversation with another person, named {counterpart}. "
    "Here is a transcript of the dialogue so far:"
    "\n\n{history}\n\n"
    "It is now your turn to respond. "
    "What is your response? Answer in the first person, "
    # "addressing {counterpart} directly, "
    "provide only your response, "
    "and keep your response SHORT, 2-3 sentences. "
    "You are addressing {counterpart}."
)