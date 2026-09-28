# =============================================================================
# MODULE 2 | LAB 2.3
# File: 03_lightfm_cold_start.py
# Purpose: LightFM Hybrid Recommender with WARP Loss and Item Features
#          Measure Cold-Start Improvement over Lab 2.2 Baseline
# Saras AI Institute | Build Predictive Models & Modern Recommenders
# =============================================================================

import os
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

# Supports the Lab 2.2 artifact structure where metrics are nested in cf_results.
cf_precision = cf_artifacts.get(
    'cf_precision',
    cf_artifacts.get('cf_results', {}).get('precision_at_10', 0.0)
)

print(f"    CF Precision@10 (Lab 2.2) : {cf_precision:.4f}  <- baseline to beat")

purchases = events[
    events['event'] == 'transaction'
][['visitorid', 'itemid']]

all_interactions = events[['visitorid', 'itemid', 'event']].copy()


# ---------------------------------------------------------------------------
# SECTION 2: Load Item Features
# ---------------------------------------------------------------------------
print("\n[2] Loading item features for LightFM...")

props1 = pd.read_csv("data/item_properties_part1.csv")
props2 = pd.read_csv("data/item_properties_part2.csv")
props = pd.concat([props1, props2], ignore_index=True)

# Keep the most recent record per item-property pair
props_latest = (
    props.sort_values('timestamp', ascending=False)
    .drop_duplicates(subset=['itemid', 'property'])
    .copy()
)

print("    Building item feature tuples...")

# TODO: Create structural item features combining property names and string values
# Hint: Assign a 'feature' column using props_latest['property'] + '_' + props_latest['value'].astype(str)
# Then groupby 'itemid' and map features into lists of strings.
props_latest['feature'] = (
    props_latest['property'].fillna('unknown_property').astype(str)
    + '_'
    + props_latest['value'].fillna('unknown_value').astype(str)
)

item_features_raw = (
    props_latest
    .groupby('itemid')['feature']
    .apply(lambda features: list(set(features)))
    .to_dict()
)

print(f"    Items with features: {len(item_features_raw):,}")


# ---------------------------------------------------------------------------
# SECTION 3: Build LightFM Dataset
# ---------------------------------------------------------------------------
print("\n[3] Building LightFM dataset...")

# TODO: Instantiate a LightFM Dataset() wrapper object
dataset = Dataset()

# TODO: Extract structural arrays capturing unique users, items, and feature strings
all_users = all_interactions['visitorid'].unique()
all_items = all_interactions['itemid'].unique()

all_features = sorted(
    {
        feature
        for feature_list in item_features_raw.values()
        for feature in feature_list
    }
)

# TODO: Fit the dataset container registry to map identifiers into internally consistent matrix indices
# Hint: Call dataset.fit(users=..., items=..., item_features=...)
dataset.fit(
    users=all_users,
    items=all_items,
    item_features=all_features
)

print(f"    Users registered  : {len(all_users):,}")
print(f"    Items registered  : {len(all_items):,}")


# ---------------------------------------------------------------------------
# SECTION 4: Build Interaction Matrix and Feature Matrix
# ---------------------------------------------------------------------------
print("\n[4] Building interaction and feature matrices...")

event_weights = {'view': 1, 'addtocart': 2, 'transaction': 3}

interactions_weighted = all_interactions.copy()
interactions_weighted['weight'] = (
    interactions_weighted['event']
    .map(event_weights)
    .fillna(0)
)

# Aggregate repeated actions into one weighted user-item interaction.
interaction_strengths = (
    interactions_weighted
    .groupby(['visitorid', 'itemid'], as_index=False)['weight']
    .sum()
)

# TODO: Build the primary user-item interaction and weights coordinate arrays
# Hint: Use dataset.build_interactions() passing an iterable of (visitorid, itemid, weight) tuples
interactions_matrix, weights_matrix = dataset.build_interactions(
    (
        (row.visitorid, row.itemid, row.weight)
        for row in interaction_strengths.itertuples(index=False)
    )
)

# TODO: Build the sparse item feature coordinate lookup matrix
# Hint: Use dataset.build_item_features() passing an iterable list of (itemid, [feature_list]) tuples
# Ensure you filter elements down to registered_items only to avoid index out-of-bounds mismatches
registered_items = set(all_items)

item_features_matrix = dataset.build_item_features(
    (
        (item_id, features)
        for item_id, features in item_features_raw.items()
        if item_id in registered_items
    )
)

print(f"    Interaction matrix shape: {interactions_matrix.shape}")
print(f"    Item feature matrix shape: {item_features_matrix.shape}")


# ---------------------------------------------------------------------------
# SECTION 5: Train / Test Split
# ---------------------------------------------------------------------------
print("\n[5] Temporal train/test split...")

events['datetime'] = pd.to_datetime(events['timestamp'], unit='ms')

# TODO: Calculate an 80% quantile temporal threshold border string across the 'datetime' axis to define a split
cutoff_date = events['datetime'].quantile(0.80)

print(f"    Cutoff date : {cutoff_date}")

events_weighted = events[['visitorid', 'itemid', 'event', 'datetime']].copy()
events_weighted['weight'] = events_weighted['event'].map(event_weights).fillna(0)

# TODO: Partition interactions_weighted into train_events (<= cutoff) and test_events (> cutoff and event == 'transaction')
train_events = events_weighted[
    events_weighted['datetime'] <= cutoff_date
].copy()

test_events = events_weighted[
    (events_weighted['datetime'] > cutoff_date) &
    (events_weighted['event'] == 'transaction')
].copy()

# Aggregate events to avoid duplicate coordinate entries.
train_strengths = (
    train_events
    .groupby(['visitorid', 'itemid'], as_index=False)['weight']
    .sum()
)

test_strengths = (
    test_events
    .groupby(['visitorid', 'itemid'], as_index=False)['weight']
    .sum()
)

# TODO: Compile train_matrix and test_matrix structures using your instantiated dataset.build_interactions helper
train_matrix, _ = dataset.build_interactions(
    (
        (row.visitorid, row.itemid, row.weight)
        for row in train_strengths.itertuples(index=False)
    )
)

test_matrix, _ = dataset.build_interactions(
    (
        (row.visitorid, row.itemid, row.weight)
        for row in test_strengths.itertuples(index=False)
    )
)

print(f"    Train interactions: {train_matrix.nnz:,}")
print(f"    Test transactions : {test_matrix.nnz:,}")


# ---------------------------------------------------------------------------
# SECTION 6: Train LightFM — Pure CF (No Features)
# ---------------------------------------------------------------------------
print("\n[6] Training LightFM — Pure CF mode (no item features)...")

# TODO: Initialize a LightFM model to test Collaborative Filtering behavior
# Hyperparameters: no_components=64, loss='warp', learning_rate=0.05, item_alpha=1e-6, user_alpha=1e-6, random_state=42
model_cf = LightFM(
    no_components=64,
    loss='warp',
    learning_rate=0.05,
    item_alpha=1e-6,
    user_alpha=1e-6,
    random_state=42
)

# TODO: Fit model_cf onto your train_matrix using 20 training epochs and num_threads=4
model_cf.fit(
    train_matrix,
    epochs=20,
    num_threads=4,
    verbose=False
)

# TODO: Calculate mean metric scores across test and train matrices using LightFM's integrated precision_at_k function
# Hint: Remember to supply your train_interactions=train_matrix constraint when calculating test precision to exclude training hits
cf_train_precision = precision_at_k(
    model_cf,
    train_matrix,
    k=10,
    num_threads=4
).mean()

cf_test_precision = precision_at_k(
    model_cf,
    test_matrix,
    train_interactions=train_matrix,
    k=10,
    num_threads=4
).mean()

print(f"    LightFM CF Train Precision@10 : {cf_train_precision:.4f}")
print(f"    LightFM CF Test  Precision@10 : {cf_test_precision:.4f}")


# ---------------------------------------------------------------------------
# SECTION 7: Train LightFM — Hybrid (CF + Item Features)
# ---------------------------------------------------------------------------
print("\n[7] Training LightFM — Hybrid mode (CF + item features)...")

# TODO: Initialize an identical hyperparameter configuration for your hybrid runner
model_hybrid = LightFM(
    no_components=64,
    loss='warp',
    learning_rate=0.05,
    item_alpha=1e-6,
    user_alpha=1e-6,
    random_state=42
)

# TODO: Fit your hybrid model on train_matrix while explicitly providing item_features=item_features_matrix
model_hybrid.fit(
    train_matrix,
    item_features=item_features_matrix,
    epochs=20,
    num_threads=4,
    verbose=False
)

# TODO: Evaluate performance values tracking precision_at_k(..., k=10) with your added item_features_matrix maps
hybrid_train_precision = precision_at_k(
    model_hybrid,
    train_matrix,
    item_features=item_features_matrix,
    k=10,
    num_threads=4
).mean()

hybrid_test_precision = precision_at_k(
    model_hybrid,
    test_matrix,
    train_interactions=train_matrix,
    item_features=item_features_matrix,
    k=10,
    num_threads=4
).mean()

print(f"    LightFM Hybrid Train Precision@10 : {hybrid_train_precision:.4f}")
print(f"    LightFM Hybrid Test  Precision@10 : {hybrid_test_precision:.4f}")


# ---------------------------------------------------------------------------
# SECTION 8: Cold-Start Improvement Demonstration
# ---------------------------------------------------------------------------
print("\n[8] Measuring cold-start improvement...")

# TODO: Extract user ID sets across test and train partitions to pinpoint users with zero interaction histories
# Hint: Subtract train visitorid sets from test visitorid sets
train_users = set(train_events['visitorid'].unique())
test_users = set(test_events['visitorid'].unique())
cold_start_users = test_users - train_users

print(f"    Cold-start users       : {len(cold_start_users):,}")


# ---------------------------------------------------------------------------
# SECTION 9: Full Comparison Table
# ---------------------------------------------------------------------------
print("\n[9] FULL MODEL COMPARISON:")
print(f"\n    {'Model':<40} {'Precision@10':>12}")
print(f"    {'-' * 55}")
print(f"    {'Item-Item CF (Lab 2.2)':<40} {cf_precision:>12.4f}")
print(f"    {'LightFM Pure CF (no features)':<40} {cf_test_precision:>12.4f}")
print(f"    {'LightFM Hybrid (CF + item features)':<40} {hybrid_test_precision:>12.4f}")


# ---------------------------------------------------------------------------
# SECTION 10: Training Progression Visualization
# ---------------------------------------------------------------------------
print("\n[10] Plotting training progression...")

os.makedirs("output", exist_ok=True)

cf_epochs = []
hybrid_epochs = []

# TODO: Re-instantiate separate progress tracking estimators matching your hyperparameter configs
model_cf_prog = LightFM(
    no_components=64,
    loss='warp',
    learning_rate=0.05,
    item_alpha=1e-6,
    user_alpha=1e-6,
    random_state=42
)

model_hybrid_prog = LightFM(
    no_components=64,
    loss='warp',
    learning_rate=0.05,
    item_alpha=1e-6,
    user_alpha=1e-6,
    random_state=42
)

for epoch in range(1, 21):
    # TODO: Perform single incremental training steps using .fit_partial() across each epoch loop
    # Ensure item_features are supplied to the hybrid progress model instance
    model_cf_prog.fit_partial(
        train_matrix,
        epochs=1,
        num_threads=4
    )

    model_hybrid_prog.fit_partial(
        train_matrix,
        item_features=item_features_matrix,
        epochs=1,
        num_threads=4
    )

    # TODO: Calculate evaluation outputs for each model slice at the current epoch step and append results to metrics lists
    cf_p = precision_at_k(
        model_cf_prog,
        test_matrix,
        train_interactions=train_matrix,
        k=10,
        num_threads=4
    ).mean()

    h_p = precision_at_k(
        model_hybrid_prog,
        test_matrix,
        train_interactions=train_matrix,
        item_features=item_features_matrix,
        k=10,
        num_threads=4
    ).mean()

    cf_epochs.append(cf_p)
    hybrid_epochs.append(h_p)

# --- Generate Step Tracking Evaluation Plots ---
fig, ax = plt.subplots(figsize=(10, 5))

# TODO: Overlay line traces plotting cf_epochs and hybrid_epochs performance metrics using ax.plot()
ax.plot(
    range(1, 21),
    cf_epochs,
    marker='o',
    linewidth=2,
    color='#4C72B0',
    label='LightFM Pure CF'
)

ax.plot(
    range(1, 21),
    hybrid_epochs,
    marker='s',
    linewidth=2,
    color='#55A868',
    label='LightFM Hybrid'
)

ax.set_title(
    "Lab 2.3: LightFM Training Progression\n"
    "Hybrid vs Pure CF — Precision@10 per Epoch",
    fontweight='bold'
)
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
lightfm_artifacts = {
    'dataset': dataset,
    'model_cf': model_cf,
    'model_hybrid': model_hybrid,
    'interactions_matrix': interactions_matrix,
    'weights_matrix': weights_matrix,
    'train_matrix': train_matrix,
    'test_matrix': test_matrix,
    'item_features_matrix': item_features_matrix,
    'all_users': all_users,
    'all_items': all_items,
    'all_features': all_features,
    'cold_start_users': cold_start_users,
    'cf_train_precision_at_10': cf_train_precision,
    'cf_test_precision_at_10': cf_test_precision,
    'hybrid_train_precision_at_10': hybrid_train_precision,
    'hybrid_test_precision_at_10': hybrid_test_precision,
    'cf_epochs': cf_epochs,
    'hybrid_epochs': hybrid_epochs
}

with open("data/lightfm_artifacts.pkl", "wb") as f:
    pickle.dump(lightfm_artifacts, f)

print("\n    Saved -> data/lightfm_artifacts.pkl")
print("    Move to: 04_evaluation_comparison.py")
