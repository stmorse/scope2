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
    
    "How likely are you "
    "to agree with the following statement: "
    "{hypothesis}\n\n"
    
    "Give your answer as an integer between 1 and 9, following this rubric: "
    
    "1 = strongly disagree\n"
    "2 = disagree\n"
    "3 = mostly disagree\n"
    "4 = slightly disagree\n"
    "5 = neutral\n"
    "6 = slightly agree\n"
    "7 = mostly agree\n"
    "8 = agree\n"
    "9 = strongly agree\n"
    
    "Ensure your rating faithfully represents your utterances in the dialogue. "
    "Provide your response as a single integer, nothing else."
)

INTERVIEW_SUMMARY = (
    # "You are having a conversation with another person, named {counterpart}. "
    # "Here is a transcript of the dialogue so far:"
    # "\n\n{history}\n\n"
    # "You were asked how likely now you were "
    # "to agree with the following statement: "
    # "{hypothesis}\n\n"
    # "You gave your answer as an integer between 1 and 10, "
    # "with 1 representing absolutely no agreement, and 10 "
    # "representing complete, unequivocal agreement. "
    # "You answered: {rating}."
    # "Now you have the chance to explain yourself. "
    # "Why did you give this rating?"
    
    "Consider the following explanation:\n"
    "{explanation}\n"
    "Given this lengthy rationale and rating consisting "
    "of a number between 1 and 10 "
    "(with 1 representing absolutely no agreement, and 10 "
    "representing complete, unequivocal agreement), "
    "summarize it by providing just the single integer "
    "representing the rating, no other response."
)