import json
import pickle

from agent.llm_client import LLMClient

SCENARIO_NAME = "russia"

PERSONA_PROMPT = (
    "I need you to generate a description of someone's stance that captures a certain "
    "valence score toward a certain position. Valence is a float between "
    "-1.0 and 1.0, with -1.0 indicating complete disagreement with "
    "the position, 0.0 indicating a neutral stance, and 1.0 indicating "
    "complete agreement with the position.\n\n"
    
    "The position is: {hypothesis}.\n\n"
    "The desired valence is: {valence}.\n\n"

    "Write the description in the third person but avoiding pronouns "
    "(for example, instead of \"He maintains ...\" or \"She endorses ...\","
    "use \"Maintains ...\" or \"Endorses ...\"). "
    "Limit the description to exactly ONE sentence."

)

client = LLMClient(provider="ollama", model="llama3.3:latest")

# load scenario
with open(f"scenarios/{SCENARIO_NAME}.json", "r") as f:
    scenario = json.load(f)

hypothesis = scenario["base"]

for valence in [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]:
    prompt = PERSONA_PROMPT.format(
        hypothesis=hypothesis, valence=f"{valence:.1f}")
    response = client.get_response(prompt)
    print(f"\nVALENCE: {valence:.1f}\nRESPONSE: {response}\n")

print("\nCOMPLETE")

