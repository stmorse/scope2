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

INTERVIEW = (
    "You are having a conversation with another person, named {counterpart}. "
    "Here is a transcript of the dialogue so far:"
    "\n\n{history}\n\n"
    "Based on this dialogue and your persona, how likely are you "
    "to agree with the following statement: "
    "{hypothesis}\n\n"
    "Give your answer as an integer between 1 and 10, "
    "with 1 representing absolutely no agreement, and 10 "
    "representing complete, unequivocal agreement. "
    "Respond with just the number, nothing else."
)

INTERVIEW_RATIONALE = (
    "You are having a conversation with another person, named {counterpart}. "
    "Here is a transcript of the dialogue so far:"
    "\n\n{history}\n\n"
    "Based on this dialogue and your persona, "
    "you were asked how likely you were "
    "to agree with the following statement: "
    "{hypothesis}\n\n"
    "You gave your answer as an integer between 1 and 10, "
    "with 1 representing absolutely no agreement, and 10 "
    "representing complete, unequivocal agreement. "
    "You answered: {rating}."
    "Now you have the chance to explain yourself. "
    "Why did you give this rating?"
)