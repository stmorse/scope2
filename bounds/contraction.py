"""
Estimate local contraction c for one-step conversation dynamics.

We compare two nearby starting states z, z' (encoded by SentenceTransformers),
apply the SAME action u (assistant message) to both, sample N user replies
from your Partner Agent (Agent "0"), roll up to next states, and measure:
    r = W1( K_u(z), K_u(z') ) / d(z, z')
where:
  - d is cosine distance on L2-normalized embeddings,
  - W1 is approximated with sliced Wasserstein-1 (SWD).

Outputs:
  - contraction_summary.json
  - pair_metrics.csv
  - (optional) PNG plots if matplotlib is available

Adjust the import path for Agent / ConversationState below.
"""

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Iterable, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from agent.agent import Agent
from mcts.mcts_node import ConversationState


# ------------------------------- Embedding ---------------------------------- #

class StateEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
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


def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    # Inputs assumed L2-normalized
    return float(1.0 - np.clip(np.dot(u, v), -1.0, 1.0))


# ---------------------------- Sliced Wasserstein ----------------------------- #

def sliced_wasserstein_1(
    X: np.ndarray, Y: np.ndarray, num_projections: int = 256, rng: Optional[np.random.RandomState] = None
) -> float:
    """
    SWD between two point clouds X, Y in R^d, using equal weights.
    Requires X.shape[0] == Y.shape[0] (use same N samples for both clouds).
    """
    assert X.ndim == 2 and Y.ndim == 2 and X.shape[1] == Y.shape[1]
    assert X.shape[0] == Y.shape[0], "Use equal N samples for both clouds for this SWD implementation."

    if rng is None:
        rng = np.random.RandomState(42)
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


# ------------------------------- Utilities ---------------------------------- #

def append_turn(state: ConversationState, text: str) -> ConversationState:
    """Return a NEW state with one message appended."""
    # return ConversationState(messages=state.messages + [text])
    return state.add_message(text)

def last_speaker_is_user(state: ConversationState) -> bool:
    # By convention: index 0=user, 1=assistant, 2=user, 3=assistant, ...
    return (len(state.messages) % 2) == 1  # if last index is even=>user; but len-1 even <=> len odd
    # More explicit:
    # last_idx = len(state.messages) - 1
    # return last_idx % 2 == 0

def ensure_user_turn_end(state: ConversationState) -> ConversationState:
    """Ensure the last message is from the user (Agent 0). If not, add a neutral user line."""
    if len(state.messages) == 0:
        return ConversationState(messages=["Hi there."])  # start with a user
    last_idx = len(state.messages) - 1
    if last_idx % 2 == 0:  # already user
        return state
    # Append a neutral user filler to make user the last speaker
    return append_turn(state, "Okay—one more thought from me.")


# -------------------------- Pair generation (local) ------------------------- #

NEUTRAL_FILLERS_USER = [
    "Hmm, I see.",
    "Okay, that makes sense.",
    "Right, go on.",
    "Understood.",
    "I follow."
]

LEXICAL_SWAPS = {
    "good": ["great", "solid"],
    "bad": ["poor", "not ideal"],
    "fast": ["quick", "rapid"],
    "slow": ["sluggish", "gradual"],
    "help": ["assist", "support"],
    "idea": ["notion", "thought"],
    "important": ["key", "crucial"],
}

def lexical_tweak(s: str, rng: random.Random) -> str:
    words = s.split()
    indices = list(range(len(words)))
    rng.shuffle(indices)
    for idx in indices:
        w = words[idx].strip(",.!?;:").lower()
        if w in LEXICAL_SWAPS:
            repl = rng.choice(LEXICAL_SWAPS[w])
            words[idx] = repl
            return " ".join(words)
    # if nothing swapped, append a mild intensifier
    return s + " (slightly rephrased)."

def make_nearby_variant(state: ConversationState, rng: random.Random) -> ConversationState:
    """
    Produce a nearby state by minimally editing the last USER message (or inserting a neutral user filler).
    Assumes last message is user after ensure_user_turn_end().
    """
    s = state.messages.copy()
    if len(s) == 0:
        return ConversationState(messages=["Hi there."])

    last_user_idx = len(s) - 1  # ensured to be user
    last_user_msg = s[last_user_idx]

    mode = rng.choice(["lexical", "filler"])
    if mode == "lexical":
        s[last_user_idx] = lexical_tweak(last_user_msg, rng)
    else:
        # Insert a neutral filler BEFORE the last assistant next time; here we just extend user with a short clause
        s[last_user_idx] = last_user_msg + " " + rng.choice(NEUTRAL_FILLERS_USER)

    return ConversationState(messages=s)


# ----------------------------- Core Experiment ------------------------------ #

ACTIONS = [
    "Could you tell me more about that?",
    "What part of this matters most to you?",
    "Let's switch gears: what's a related topic you'd like to explore?"
]

@dataclass
class ContractionConfig:
    M_pairs: int = 200
    N_samples: int = 100
    K_projections: int = 256
    eps_min: float = 0.02
    eps_max: float = 0.06
    action_idx: int = 0
    seed: int = 1337
    device: Optional[str] = None
    truncate_history: int = 12  # keep last k turns for embedding stability


def truncate_state(state: ConversationState, k: int) -> ConversationState:
    if k is None or k <= 0:
        return state
    msgs = state.messages[-k:]
    return ConversationState(messages=msgs)


def sample_partner_replies(
    partner: Agent, state_with_action: ConversationState, N: int
) -> List[str]:
    """Sample N user replies from the partner agent given the (state + assistant action)."""
    replies = []
    for _ in range(N):
        y = partner.get_response(state_with_action)
        replies.append(y)
    return replies


def run_contraction_estimate(
    embedder: StateEmbedder,
    assistant_action: str,
    partner: Agent,
    seed_states: List[ConversationState],
    cfg: ContractionConfig,
) -> Tuple[dict, List[dict]]:
    """
    Returns (summary_json, rows) where rows contain per-pair metrics.
    """
    rng_np = np.random.RandomState(cfg.seed)
    rng_py = random.Random(cfg.seed)

    # Prepare starting pairs
    pairs: List[Tuple[ConversationState, ConversationState]] = []
    for s in seed_states:
        s0 = ensure_user_turn_end(truncate_state(s, cfg.truncate_history))
        s1 = ensure_user_turn_end(truncate_state(make_nearby_variant(s0, rng_py), cfg.truncate_history))
        z0 = embedder.embed_state(s0)
        z1 = embedder.embed_state(s1)
        d0 = cosine_distance(z0, z1)
        if cfg.eps_min <= d0 <= cfg.eps_max:
            pairs.append((s0, s1))
        if len(pairs) >= cfg.M_pairs:
            break

    if len(pairs) == 0:
        raise RuntimeError("No nearby pairs found in the specified epsilon range. Loosen eps_min/max or adjust variants.")

    # Main loop
    rows = []
    ratios = []

    for idx, (s, s_prime) in enumerate(pairs, 1):
        z = embedder.embed_state(s)
        z_prime = embedder.embed_state(s_prime)
        d0 = cosine_distance(z, z_prime)

        # Append SAME assistant action u to both
        s_u = append_turn(s, assistant_action)            # assistant turn
        s_prime_u = append_turn(s_prime, assistant_action)

        # Sample N user replies for each condition
        Ys = sample_partner_replies(partner, s_u, cfg.N_samples)
        Ys_prime = sample_partner_replies(partner, s_prime_u, cfg.N_samples)

        # Roll up to next states and embed
        Z1 = []
        for y in Ys:
            next_state = append_turn(s_u, y)              # user reply appended
            v = embedder.embed_state(next_state)
            Z1.append(v)
        Z1 = np.stack(Z1, axis=0)

        Z1_prime = []
        for y in Ys_prime:
            next_state_p = append_turn(s_prime_u, y)
            v = embedder.embed_state(next_state_p)
            Z1_prime.append(v)
        Z1_prime = np.stack(Z1_prime, axis=0)

        # SWD between clouds
        W = sliced_wasserstein_1(Z1, Z1_prime, num_projections=cfg.K_projections, rng=rng_np)
        r = W / max(d0, 1e-8)
        ratios.append(r)

        rows.append({
            "pair_id": idx,
            "d0": d0,
            "W": W,
            "r": r,
            "n_samples": cfg.N_samples,
        })

    # Aggregate summary
    r_arr = np.array(ratios, dtype=np.float64)
    summary = {
        "M_pairs": len(pairs),
        "N_samples": cfg.N_samples,
        "K_projections": cfg.K_projections,
        "eps_range": [cfg.eps_min, cfg.eps_max],
        "action": assistant_action,
        "r_median": float(np.median(r_arr)),
        "r_mean": float(np.mean(r_arr)),
        "r_p90": float(np.quantile(r_arr, 0.90)),
        "r_p95": float(np.quantile(r_arr, 0.95)),
        "r_max": float(np.max(r_arr)),
    }
    return summary, rows


# ---------------------------------- CLI ------------------------------------- #

DEFAULT_SEEDS = [
    ConversationState(messages=[
        "I'm trying to improve my 5k time; I'm stuck around 24 minutes.",
        "What's your current weekly mileage and any speed work?",
    ]),
    ConversationState(messages=[
        "I'm considering switching careers into data science from finance.",
        "What attracts you to data science, and what experience do you already have?",
    ]),
    ConversationState(messages=[
        "I'm worried my child is spending too much time on social media.",
        "What changes have you noticed, and what have you tried so far?",
    ]),
    ConversationState(messages=[
        "I'm interested in the ethics of AI and how to avoid harm.",
        "Are there particular scenarios or domains you’re most concerned about?",
    ]),
    ConversationState(messages=[
        "I feel burned out at work and I'm not sure what to do.",
        "What parts of your work drain you most, and what energizes you?",
    ]),
]

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate local contraction for one-step conversation dynamics.")
    p.add_argument("--M", type=int, default=200, help="Number of nearby pairs to evaluate.")
    p.add_argument("--N", type=int, default=100, help="Samples per state (user replies).")
    p.add_argument("--K", type=int, default=256, help="Sliced Wasserstein projections.")
    p.add_argument("--eps-min", type=float, default=0.02, help="Min starting cosine distance for pair selection.")
    p.add_argument("--eps-max", type=float, default=0.06, help="Max starting cosine distance for pair selection.")
    p.add_argument("--action-idx", type=int, default=0, help="Index into the ACTIONS list.")
    p.add_argument("--seed", type=int, default=1337, help="Random seed.")
    p.add_argument("--device", type=str, default=None, help="Torch device for embeddings (e.g., 'cpu', 'cuda').")
    p.add_argument("--out-dir", type=str, default="contraction_out", help="Directory for outputs.")
    p.add_argument("--use-default-seeds", action="store_true", help="Use built-in seed states.")
    p.add_argument("--seeds-json", type=str, default=None,
                   help="Path to a JSON file with a list of conversations; each is a list of message strings.")
    return p.parse_args()


def load_seed_states(args: argparse.Namespace) -> List[ConversationState]:
    if args.use_default_seeds:
        return DEFAULT_SEEDS

    if args.seeds_json:
        with open(args.seeds_json, "r", encoding="utf-8") as f:
            conversations = json.load(f)
        states = [ConversationState(messages=conv) for conv in conversations]
        return states

    # Fallback: use defaults
    return DEFAULT_SEEDS


# def maybe_plot(rows: List[dict], out_dir: str):
#     if not HAVE_PLT:
#         return
#     os.makedirs(out_dir, exist_ok=True)
#     r = np.array([row["r"] for row in rows], dtype=np.float64)
#     d0 = np.array([row["d0"] for row in rows], dtype=np.float64)

#     plt.figure()
#     plt.hist(r, bins=30)
#     plt.xlabel("r = W / d0")
#     plt.ylabel("count")
#     plt.title("Local contraction ratios")
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, "hist_r.png"), dpi=160)

#     plt.figure()
#     plt.scatter(d0, r, s=10, alpha=0.7)
#     plt.xlabel("d0 (cosine distance between starting states)")
#     plt.ylabel("r = W / d0")
#     plt.title("r vs d0")
#     plt.tight_layout()
#     plt.savefig(os.path.join(out_dir, "scatter_r_vs_d0.png"), dpi=160)


def main():
    args = parse_args()
    cfg = ContractionConfig(
        M_pairs=args.M,
        N_samples=args.N,
        K_projections=args.K,
        eps_min=args.eps_min,
        eps_max=args.eps_max,
        action_idx=args.action_idx,
        seed=args.seed,
        device=args.device,
    )

    # Instantiate your Agents:
    # - partner acts as the USER ("Agent 0") producing replies to assistant actions
    partner = Agent(name="Agent 0", provider="ollama", model="llama3.3:latest")

    # Embedder
    embedder = StateEmbedder(device=cfg.device)

    # Load seeds
    seed_states = load_seed_states(args)

    # Choose the action (assistant utterance)
    if not (0 <= cfg.action_idx < len(ACTIONS)):
        raise ValueError(f"--action-idx out of range (0..{len(ACTIONS)-1})")
    assistant_action = ACTIONS[cfg.action_idx]

    # Run
    summary, rows = run_contraction_estimate(
        embedder=embedder,
        assistant_action=assistant_action,
        partner=partner,
        seed_states=seed_states,
        cfg=cfg,
    )

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "contraction_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.out_dir, "pair_metrics.csv"), "w", encoding="utf-8") as f:
        f.write("pair_id,d0,W,r,n_samples\n")
        for row in rows:
            f.write(f"{row['pair_id']},{row['d0']:.6f},{row['W']:.6f},{row['r']:.6f},{row['n_samples']}\n")

    # maybe_plot(rows, args.out_dir)

    # Console summary
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
