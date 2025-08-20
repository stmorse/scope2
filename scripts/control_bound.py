"""
B = sup_{z,u,u'} W(K_u(z), K_u'(z))
"""

import pickle
import sys
import os
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from agent.agent import Agent
from agent.llm_client import LLMClient
from mcts.mcts_node import ConversationState

# --- PARAMS ---

DEVICE = "cuda"
SAVE_SUFFIX = "test2"

N_Z0      = 20  # num starting sentences
N_ACTIONS = 10  # num assistant actions (reply to query)
N_SAMPLES = 30  # num user replies to each action

Z0_PROMPT = (
    "Generate {num_sentences} different questions to start a conversation. "
    "Each question should be short and all must be on different topics. "
    "Separate each question with two newlines. Do not provide anything "
    "other than your response."
)


# --- HELPER FUNCTIONS ---

class StateEmbedder:
    def __init__(self, 
            model_name: str = "all-MiniLM-L6-v2", 
            device: Optional[str] = DEVICE
    ):
        self.model = SentenceTransformer(model_name, device=device)

    @staticmethod
    def _render(state: ConversationState) -> str:
        """Render a state as tagged lines to stabilize the encoder."""
        lines = []
        for i, msg in enumerate(state.messages):
            role = "[USR]" if i % 2 == 0 else "[AST]"
            lines.append(f"{role} {msg}")
        return "\n".join(lines)

    def embed_state(self, state: ConversationState) -> np.ndarray:
        text = self._render(state)
        v = self.model.encode([text], normalize_embeddings=True)[0]  # L2-normalized
        return v.astype(np.float32)

def sliced_wasserstein_1(
    X: np.ndarray, 
    Y: np.ndarray, 
    num_projections: int = 256, 
    rng: Optional[np.random.RandomState] = None
) -> float:
    """
    SWD between two point clouds X, Y in R^d, using equal weights.
    Requires X.shape[0] == Y.shape[0] (use same N samples for both clouds).
    """
    assert X.ndim == 2 and Y.ndim == 2 and X.shape[1] == Y.shape[1]
    assert X.shape[0] == Y.shape[0], "Use equal N samples for both clouds for this SWD implementation."

    if rng is None:
        rng = np.random.RandomState(314)
    d = X.shape[1]
    total = 0.0

    for _ in range(num_projections):
        # random unit direction
        w = rng.normal(size=(d,))
        nrm = np.linalg.norm(w)
        if nrm == 0:
            continue
        w /= nrm
        proj_X = X @ w
        proj_Y = Y @ w
        proj_X.sort()
        proj_Y.sort()
        total += float(np.mean(np.abs(proj_X - proj_Y)))

    return total / num_projections


# --- MAIN SCRIPT ---

def main():

    # --- LOAD EMBEDDING MODEL ---

    print(f"Loading embedding model (Device: {DEVICE})")
    embedder = StateEmbedder()

    
    # --- INIT AGENTS ---

    print(f"\nInitializing agents ...\n")
    agents = {i: Agent(
        name=f"Agent {i}",
        provider="ollama", 
        model="llama3.3:latest", 
        personality="(None specified)",
        forcing=False
    ) for i in range(2)}


    # --- generate starting sentences ---
    
    print(f"Generating {N_Z0} starting sentences...")
    client = LLMClient(provider="ollama", model="llama3.3:latest")
    response = client.get_response(Z0_PROMPT.format(num_sentences=N_Z0))
    z0_sentences = [s.strip() for s in response.split("\n") if len(s) > 5]

    print(f"\nZ0:\n{"\n".join(z0_sentences)}\n")


    # --- compute estimated B for this z0 and multiple u, u' ---
    output = []  # list of dicts
    for z0_sentence in z0_sentences:
        print(f"\n\n{"="*60}\nWorking Z0: {z0_sentence}\n")
        
        # will hold all results for this z0
        # {"u": A1 action, "z0": z0 embedding (repeated), "uX": z0 with u embedding
        #  "responses": ["A0 response 1", "A0 response 2", ...], 
        #  "X": nd.array of embeddings of responses}
        results = []
        
        z0 = ConversationState(messages=[z0_sentence])
        z0_embedding = embedder.embed_state(z0)

        # generate an action, sample many possible Agent 0 responses
        results = []   # list of dicts
        for i in range(N_ACTIONS):
            print(f"\nACTION {i}")

            # "u": A1 action, 
            # "u_embed": z0 + u embedding
            # "responses": [all user responses]
            # "X": nd.array of embeddings of full state (z0 + u + response)
            result = {}

            u = agents[1].get_response(z0)
            zc = z0.add_message(u)
            zc_embedding = embedder.embed_state(zc)
            result["u_embedding"] = zc_embedding.copy()
            result["u"] = u
            
            print(f" > Action: \"{u}\"")

            # TODO: this is doing one by one, should do in batch (embedding)
            responses = []
            embeddings = []
            for j in range(N_SAMPLES):
                a0 = agents[0].get_response(zc)
                responses.append(a0)
                rc = zc.add_message(a0)
                embeddings.append(embedder.embed_state(rc).copy())
                print(f"   - Response {j}: \"{a0}\"")

            # store and embed
            result["response"] = responses
            result["X"] = np.vstack(embeddings)

            results.append(result)

        # compute all the distances between X's
        print("\n\nComputing distances...")
        distances = np.zeros((N_ACTIONS, N_ACTIONS))
        for i in range(N_ACTIONS):
            print(f" > Row {i}/{N_ACTIONS-1}")
            for j in range(i+1, N_ACTIONS):
                X = results[i]["X"]
                Y = results[j]["X"]
                d = sliced_wasserstein_1(X, Y)
                distances[i,j] = d

        output.append({
            "results": results,
            "distances": distances.copy(),
            "z0": z0_sentence,
            "z0_embedding": z0_embedding.copy(),
        })

    print("Saving...")
    with open(f"notebooks/b_{SAVE_SUFFIX}.pkl", "wb") as f:
        pickle.dump(output, f)

    print("\nCOMPLETE")


if __name__=="__main__":
    main()       
        

    