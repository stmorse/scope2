"""
Loads records, embeds, scores, projects to 2d, saves
"""

import pickle
import sys
import os
import time

# hack so we can import normally from other packages
parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# set matplotlib env
# os.environ["MPLCONFIGDIR"] = "/sciclone/home/stmorse/.config/matplotlib"
# import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.manifold import TSNE

from mcts.mcts_node import ConversationState
from mcts.reward_functions import SafetyReward, TopicReward

# --- VARS ---

reward = "topic"
num_candidates = 3
save_path = "notebooks/output_topic"


# --- LOAD RECORDS ---

print(f"Loading...")
t0 = time.time()

records = []
for i in range(num_candidates):
    with open(f"experiments/{reward}/turn_0_candidate_{i}.pkl", "rb") as f:
        records.append(pickle.load(f))


# --- LOAD EMBEDDING MODELS ---

model = SentenceTransformer("all-MiniLM-L6-v2")

# reward_model = SafetyReward(harm=True if reward=="harm" else False)
# print(f"Reward model ({reward}): harm={True if reward=="harm" else False}")

reward_model = TopicReward(topic_sentence="Birds are surveillance devices used by the government.")


# --- EMBEDDING AND SCORING ---
embeddings = []
rewards    = [] 

max_sims = min(len(records[cand]) for cand in range(num_candidates))
print(f"Using {max_sims} simulations per candidate")

print(f"Embedding / scoring ... {time.time()-t0:.3f}")

for cand in range(num_candidates):
    for sim in range(max_sims):
        # grab all responses from rollout
        rollout = records[cand][sim]["rollout"]

        # we're going to remove "Agent1: " from each
        w = len("AgentX: ")
        
        # create, embed, and score, successive stages of the conversation
        for m in range(1, len(rollout)+1):
            # create ConversationState for this depth of the conversation
            subset = rollout[:m]
            state = ConversationState(
                messages=[r[w:] for r in subset],  
                current_turn=len(subset) % 2,      # we don't need this
                depth=len(subset)                  # or this
            )

            # create cumulative conversations
            # cumulative = state.convert_to_cumulative()
            concat = "\n".join(state.get_all_messages())

            # trim from end to fit in embedding model's context max
            concat = concat[-min(1000, len(concat)):]

            # embed this portion
            embs = model.encode(concat)
            embeddings.append(embs)

            # score this portion
            rewards.append(reward_model.calculate_reward(state))

embeddings = np.vstack(embeddings)
rewards = np.array(rewards)
print(f"Complete ({time.time()-t0:.3f})")
print(f"Embeddings shape: {embeddings.shape}")
print(f"Rewards shape: {rewards.shape}")

# Exit the script at this point
# print("Script terminated at embeddings stage.")
# sys.exit(0)

# --- SANITY CHECK ON FIRST EMBEDDINGS ---

print(f"Sanity check:")
entry0 = []

steps = 6
for k in range(num_candidates):
    for i in range(max_sims):  # omitting 0th record (check)
        a = (k*steps*max_sims) + (i*steps)
        entry0.append(embeddings[a,:])

entry0 = np.vstack(entry0)
print(entry0.shape)

tol = 1e-5  # tolerance level
# Compute the range for each column across all rows
col_range = np.ptp(entry0, axis=0)

# Check if all values in each column are within the tolerance
all_columns_close = np.all(col_range < tol)
print("Are all columns close?", all_columns_close)

# Print the indices of columns that are not close
if not all_columns_close:
    print(
        "Columns with differences exceeding tolerance:", 
        np.where(col_range >= tol)[0])


# --- LOW-DIM PROJECT ---

print(f"Projecting ... {time.time()-t0:.3f}")

e2d = TSNE(
    n_components=2, 
    perplexity=30, 
    random_state=314
).fit_transform(embeddings)

print(f"Projection: {e2d.shape}")


# --- SAVE FOR PLOTTING ---

print(f"Saving to file {save_path} ... {time.time()-t0:.3f}")
output = {
    "embeddings": embeddings,
    "rewards": rewards,
    "e2d": e2d
}
with open(f"{save_path}.pkl", "wb") as f:
    pickle.dump(output, f)


print(f"Complete. ({time.time()-t0:.3f})")


# --- PLOT ---

# fig, ax = plt.subplots(1,1, figsize=(8,8))

# colors = ['c', 'g', 'y']
# steps = 6

# for k in range(num_candidates):
#     # ax = axs[k]

#     for i in range(max_sims): 
#         # (omit first response, all identical)
#         a = (k*steps*max_sims) + (i*steps) + 1
#         b = a + steps - 1
#         z = e2d[a:b,:]

#         ax.plot(
#             z[:,0], z[:,1], 
#             color=colors[k], linestyle='-', marker='o',
#             alpha=0.7
#         )
        
#         # start / end
#         ax.scatter(z[0,0], z[0,1], s=40, c='k', zorder=100)
#         ax.scatter(z[-1,0], z[-1,1], s=40, c='r', zorder=100)

#     # Save the figure as a PDF
#     # plt.tight_layout()
#     fig.savefig(
#         f"../experiments/{reward}/{figname}.png", 
#         format='png', dpi=300, bbox_inches='tight'
#     )
#     # plt.show()