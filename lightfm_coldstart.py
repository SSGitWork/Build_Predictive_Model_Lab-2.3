# =============================================================================
# MODULE 2 | LAB 2.3
# File: 03_lightfm_cold_start.py
# Purpose: LightFM Hybrid Recommender with WARP Loss and Item Features
#          Measure Cold-Start Improvement over Lab 2.2 Baseline
# Saras AI Institute | Build Predictive Models & Modern Recommenders
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import pickle
import warnings
warnings.filterwarnings('ignore')

from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score

print("=" * 60)
print("  MODULE 2 | LAB 2.3")
print("  LightFM Hybrid Recommender")
print("  Method: WARP Loss + Item Feature Embeddings")
print("=" * 60)

# ---------------------------------------------------------------------------
# SECTION 1: Load Data and Artifacts from Previous Labs
# ---------------------------------------------------------------------------
print("\n[1] Loading data and artifacts from Labs 2.1 and 2.2...")

events = pd.read_csv("data/events.csv")

with open("data/cf_artifacts.pkl", "rb") as f:
    cf_artifacts = pickle.load(f)

cf_precision = cf_artifacts['cf_results']['precision_at_10']
print(f"    CF Precision@10 (Lab 2.2) : {cf_precision:.4f}  <- baseline to beat")

purchases = events[events['event'] == 'transaction'][['visitorid', 'itemid']]
all_interactions = events[['visitorid', 'itemid', 'event']].copy()


# ---------------------------------------------------------------------------
# SECTION 2: Load Item Features
# ---------------------------------------------------------------------------
print("\n[2] Loading item features for LightFM...")

props1 = pd.read_csv("data/item_properties_part1.csv")
props2 = pd.read_csv("data/item_properties_part2.csv")
props  = pd.concat([props1, props2], ignore_index=True)

# Keep the most recent record per item-property pair
props_latest = (
    props.sort_values('timestamp', ascending=False)
    .drop_duplicates(subset=['itemid', 'property'])
)

print("    Building item feature tuples...")
# TODO: Create structural item features combining property names and string values
# Hint: Assign a 'feature' column using props_latest['property'] + '_' + props_latest['value'].astype(str)
# Then groupby 'itemid' and map features into lists of strings.
item_features_raw = None

print(f"    Items with features: {len(item_features_raw) if item_features_raw is not None else 0:,}")


# ---------------------------------------------------------------------------
# SECTION 3: Build LightFM Dataset
# ---------------------------------------------------------------------------
print("\n[3] Building LightFM dataset...")

# TODO: Instantiate a LightFM Dataset() wrapper object
dataset = None

# TODO: Extract structural arrays capturing unique users, items, and feature strings
all_users = None
all_items = None
all_features = None

# TODO: Fit the dataset container registry to map identifiers into internally consistent matrix indices
# Hint: Call dataset.fit(users=..., items=..., item_features=...)


print(f"    Users registered  : {len(all_users) if all_users is not None else 0:,}")
print(f"    Items registered  : {len(all_items) if all_items is not None else 0:,}")


# ---------------------------------------------------------------------------
# SECTION 4: Build Interaction Matrix and Feature Matrix
# ---------------------------------------------------------------------------
print("\n[4] Building interaction and feature matrices...")

event_weights = {'view': 1, 'addtocart': 2, 'transaction': 3}
interactions_weighted = all_interactions.copy()
interactions_weighted['weight'] = interactions_weighted['event'].map(event_weights)

# TODO: Build the primary user-item interaction and weights coordinate arrays
# Hint: Use dataset.build_interactions() passing an iterable of (visitorid, itemid, weight) tuples
interactions_matrix, weights_matrix = None, None

# TODO: Build the sparse item feature coordinate lookup matrix
# Hint: Use dataset.build_item_features() passing an iterable list of (itemid, [feature_list]) tuples
# Ensure you filter elements down to registered_items only to avoid index out-of-bounds mismatches
registered_items = set(all_items) if all_items is not None else set()
item_features_matrix = None

print(f"    Interaction matrix shape: {interactions_matrix.shape if interactions_matrix is not None else 'N/A'}")
print(f"    Item feature matrix shape: {item_features_matrix.shape if item_features_matrix is not None else 'N/A'}")


# ---------------------------------------------------------------------------
# SECTION 5: Train / Test Split
# ---------------------------------------------------------------------------
print("\n[5] Temporal train/test split...")

events['datetime'] = pd.to_datetime(events['timestamp'], unit='ms')

# TODO: Calculate an 80% quantile temporal threshold border string across the 'datetime' axis to define a split
cutoff_date = None
print(f"    Cutoff date : {cutoff_date}")

# TODO: Partition interactions_weighted into train_events (<= cutoff) and test_events (> cutoff and event == 'transaction')
train_events = None
test_events = None

# TODO: Compile train_matrix and test_matrix structures using your instantiated dataset.build_interactions helper
train_matrix, _ = None, None
test_matrix, _ = None, None


# ---------------------------------------------------------------------------
# SECTION 6: Train LightFM — Pure CF (No Features)
# ---------------------------------------------------------------------------
print("\n[6] Training LightFM — Pure CF mode (no item features)...")

# TODO: Initialize a LightFM model to test Collaborative Filtering behavior
# Hyperparameters: no_components=64, loss='warp', learning_rate=0.05, item_alpha=1e-6, user_alpha=1e-6, random_state=42
model_cf = None

# TODO: Fit model_cf onto your train_matrix using 20 training epochs and num_threads=4


# TODO: Calculate mean metric scores across test and train matrices using LightFM's integrated precision_at_k function
# Hint: Remember to supply your train_interactions=train_matrix constraint when calculating test precision to exclude training hits
cf_train_precision = 0.0
cf_test_precision  = 0.0

print(f"    LightFM CF Train Precision@10 : {cf_train_precision:.4f}")
print(f"    LightFM CF Test  Precision@10 : {cf_test_precision:.4f}")


# ---------------------------------------------------------------------------
# SECTION 7: Train LightFM — Hybrid (CF + Item Features)
# ---------------------------------------------------------------------------
print("\n[7] Training LightFM — Hybrid mode (CF + item features)...")

# TODO: Initialize an identical hyperparameter configuration for your hybrid runner
model_hybrid = None

# TODO: Fit your hybrid model on train_matrix while explicitly providing item_features=item_features_matrix


# TODO: Evaluate performance values tracking precision_at_k(..., k=10) with your added item_features_matrix maps
hybrid_train_precision = 0.0
hybrid_test_precision  = 0.0

print(f"    LightFM Hybrid Train Precision@10 : {hybrid_train_precision:.4f}")
print(f"    LightFM Hybrid Test  Precision@10 : {hybrid_test_precision:.4f}")


# ---------------------------------------------------------------------------
# SECTION 8: Cold-Start Improvement Demonstration
# ---------------------------------------------------------------------------
print("\n[8] Measuring cold-start improvement...")

# TODO: Extract user ID sets across test and train partitions to pinpoint users with zero interaction histories
# Hint: Subtract train visitorid sets from test visitorid sets
cold_start_users = set()

print(f"    Cold-start users       : {len(cold_start_users):,}")


# ---------------------------------------------------------------------------
# SECTION 9: Full Comparison Table
# ---------------------------------------------------------------------------
print("\n[9] FULL MODEL COMPARISON:")
print(f"\n    {'Model':<40} {'Precision@10':>12}")
print(f"    {'-'*55}")
print(f"    {'Item-Item CF (Lab 2.2)':<40} {cf_precision:>12.4f}")
print(f"    {'LightFM Pure CF (no features)':<40} {cf_test_precision:>12.4f}")
print(f"    {'LightFM Hybrid (CF + item features)':<40} {hybrid_test_precision:>12.4f}")


# ---------------------------------------------------------------------------
# SECTION 10: Training Progression Visualization
# ---------------------------------------------------------------------------
print("\n[10] Plotting training progression...")

cf_epochs     = []
hybrid_epochs = []

# TODO: Re-instantiate separate progress tracking estimators matching your hyperparameter configs
model_cf_prog     = None
model_hybrid_prog = None

for epoch in range(1, 21):
    # TODO: Perform single incremental training steps using .fit_partial() across each epoch loop
    # Ensure item_features are supplied to the hybrid progress model instance
    pass

    # TODO: Calculate evaluation outputs for each model slice at the current epoch step and append results to metrics lists
    cf_p = 0.0
    h_p  = 0.0
    
    cf_epochs.append(cf_p)
    hybrid_epochs.append(h_p)



# --- Generate Step Tracking Evaluation Plots ---
fig, ax = plt.subplots(figsize=(10, 5))
# TODO: Overlay line traces plotting cf_epochs and hybrid_epochs performance metrics using ax.plot()


ax.set_title("Lab 2.3: LightFM Training Progression\nHybrid vs Pure CF — Precision@10 per Epoch", fontweight='bold')
ax.set_xlabel("Training Epoch")
ax.set_ylabel("Precision@10")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("output/03_lightfm_training.png", dpi=150, bbox_inches='tight')
plt.show()


# ---------------------------------------------------------------------------
# SECTION 11: Save LightFM Artifacts for Lab 2.4
# ---------------------------------------------------------------------------
# TODO: Save models, metadata datasets, and coordinate matrices to an output pickle package
# Target Path: "data/lightfm_artifacts.pkl"


print("\n    Saved -> data/lightfm_artifacts.pkl")
print("    Move to: 04_evaluation_comparison.py")
