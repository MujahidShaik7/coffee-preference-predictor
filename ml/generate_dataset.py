"""
generate_dataset.py
--------------------
Generates a synthetic-but-realistic dataset of coffee preferences.

Each row represents a person's answers to a short lifestyle/preference
questionnaire, paired with a coffee "label" (Espresso, Cappuccino, Latte,
Americano, Mocha, Cold Brew, Filter Coffee).

The labels are NOT assigned with deterministic if/else rules. Instead,
each coffee type has a "profile" (ideal feature values). For every
synthetic person we compute a similarity score between their answers and
every coffee profile, turn those scores into a probability distribution
(softmax), and sample the label from that distribution. This creates
realistic overlap and noise -- e.g. someone who mostly matches Latte might
still occasionally be labeled Cappuccino, exactly like real people whose
tastes aren't perfectly clean-cut.

Run:
    python ml/generate_dataset.py
Produces:
    data/coffee_preferences.csv
"""

import numpy as np
import pandas as pd
import os

RANDOM_SEED = 42
N_SAMPLES = 2500

rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------
# 1. Feature domains
# ---------------------------------------------------------------------
STRENGTH = ["Light", "Medium", "Strong"]
MILK = ["None", "Low", "Medium", "High"]
TEMPERATURE = ["Hot", "Iced", "Cold"]
CAFFEINE = ["Low", "Medium", "High"]
TIME_OF_DAY = ["Morning", "Afternoon", "Evening", "Night"]
FLAVOR = ["Chocolate", "Nutty", "Fruity", "Caramel", "Classic", "Vanilla"]
EXPERIENCE = ["Quick & Strong", "Smooth & Relaxing", "Sweet & Creamy", "Refreshing & Cold"]
PREVIOUS = ["Espresso", "Cappuccino", "Latte", "Americano", "Mocha", "Cold Brew", "Filter Coffee", "None"]

COFFEES = ["Espresso", "Cappuccino", "Latte", "Americano", "Mocha", "Cold Brew", "Filter Coffee"]

# Ordinal encodings used only internally to compute similarity (not saved to CSV)
STRENGTH_ORD = {"Light": 1, "Medium": 2, "Strong": 3}
MILK_ORD = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
CAFFEINE_ORD = {"Low": 1, "Medium": 2, "High": 3}

# ---------------------------------------------------------------------
# 2. Ideal "profiles" for each coffee (used to compute similarity, not
#    hardcoded into the app/API -- only used here to build training data)
# ---------------------------------------------------------------------
# Values are on comparable numeric scales:
# strength(1-3), milk(0-3), sweetness(1-5), bitterness(1-5), creaminess(1-5),
# caffeine(1-3), temperature preference (0=hot,1=iced,2=cold)
PROFILES = {
    "Espresso":      dict(strength=3, milk=0, sweetness=2, bitterness=4.5, creaminess=1, caffeine=3, temp=0),
    "Americano":      dict(strength=2.5, milk=0.3, sweetness=2, bitterness=3.5, creaminess=1.2, caffeine=3, temp=0),
    "Cappuccino":    dict(strength=2.3, milk=2.2, sweetness=2.7, bitterness=2.8, creaminess=3.3, caffeine=2.3, temp=0),
    "Latte":         dict(strength=1.8, milk=3, sweetness=3.2, bitterness=2, creaminess=4.3, caffeine=2, temp=0.2),
    "Mocha":         dict(strength=1.7, milk=2.6, sweetness=4.4, bitterness=1.8, creaminess=4.2, caffeine=2, temp=0.3),
    "Cold Brew":     dict(strength=2.4, milk=0.5, sweetness=2.2, bitterness=2.3, creaminess=1.4, caffeine=2.8, temp=1.8),
    "Filter Coffee": dict(strength=2, milk=0.8, sweetness=2, bitterness=3, creaminess=1.7, caffeine=2.2, temp=0.1),
}

FEATURE_WEIGHTS = dict(
    strength=1.3, milk=1.3, sweetness=1.0, bitterness=1.1,
    creaminess=1.1, caffeine=0.9, temp=1.6,
)

# Small "personality" nudge -- flavor/experience/time-of-day/previous choice
# slightly bias certain coffees, mirroring real-world tendencies, without
# being deterministic.
FLAVOR_BIAS = {
    "Chocolate": {"Mocha": 1.6},
    "Nutty": {"Cappuccino": 0.6, "Filter Coffee": 0.5},
    "Fruity": {"Cold Brew": 0.8, "Latte": 0.3},
    "Caramel": {"Latte": 0.9, "Mocha": 0.5},
    "Classic": {"Americano": 0.7, "Filter Coffee": 0.9, "Espresso": 0.4},
    "Vanilla": {"Latte": 0.8, "Cappuccino": 0.4},
}
EXPERIENCE_BIAS = {
    "Quick & Strong": {"Espresso": 1.4, "Americano": 0.9},
    "Smooth & Relaxing": {"Filter Coffee": 0.9, "Cappuccino": 0.6, "Latte": 0.4},
    "Sweet & Creamy": {"Mocha": 1.3, "Latte": 0.9, "Cappuccino": 0.4},
    "Refreshing & Cold": {"Cold Brew": 1.8},
}
TIME_BIAS = {
    "Morning": {"Filter Coffee": 0.4, "Cappuccino": 0.3, "Latte": 0.2},
    "Afternoon": {"Latte": 0.2, "Americano": 0.2},
    "Evening": {"Cold Brew": 0.3, "Mocha": 0.2},
    "Night": {"Cold Brew": 0.5, "Espresso": -0.3},
}
PREVIOUS_BIAS_STRENGTH = 0.9  # how much a stated previous favorite nudges the label


def softmax(x):
    x = np.array(x, dtype=float)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def sample_person():
    strength = rng.choice(STRENGTH)
    milk = rng.choice(MILK, p=[0.18, 0.22, 0.32, 0.28])
    sweetness = int(rng.integers(1, 6))
    temperature = rng.choice(TEMPERATURE, p=[0.55, 0.25, 0.20])
    bitterness = int(rng.integers(1, 6))
    creaminess = int(rng.integers(1, 6))
    caffeine = rng.choice(CAFFEINE, p=[0.2, 0.45, 0.35])
    time_of_day = rng.choice(TIME_OF_DAY, p=[0.45, 0.25, 0.2, 0.1])
    flavor = rng.choice(FLAVOR)
    experience = rng.choice(EXPERIENCE)
    previous = rng.choice(PREVIOUS, p=[0.12, 0.12, 0.14, 0.1, 0.1, 0.1, 0.1, 0.22])

    return dict(
        strength=strength, milk=milk, sweetness=sweetness, temperature=temperature,
        bitterness=bitterness, creaminess=creaminess, caffeine=caffeine,
        time_of_day=time_of_day, flavor=flavor, experience=experience, previous=previous,
    )


def temp_to_num(t):
    return {"Hot": 0, "Iced": 1, "Cold": 2}[t]


def score_person_against_profiles(person):
    p_strength = STRENGTH_ORD[person["strength"]]
    p_milk = MILK_ORD[person["milk"]]
    p_sweet = person["sweetness"]
    p_bitter = person["bitterness"]
    p_cream = person["creaminess"]
    p_caff = CAFFEINE_ORD[person["caffeine"]]
    p_temp = temp_to_num(person["temperature"])

    scores = {}
    for coffee, prof in PROFILES.items():
        dist = 0.0
        dist += FEATURE_WEIGHTS["strength"] * (p_strength - prof["strength"]) ** 2
        dist += FEATURE_WEIGHTS["milk"] * (p_milk - prof["milk"]) ** 2
        dist += FEATURE_WEIGHTS["sweetness"] * (p_sweet - prof["sweetness"]) ** 2
        dist += FEATURE_WEIGHTS["bitterness"] * (p_bitter - prof["bitterness"]) ** 2
        dist += FEATURE_WEIGHTS["creaminess"] * (p_cream - prof["creaminess"]) ** 2
        dist += FEATURE_WEIGHTS["caffeine"] * (p_caff - prof["caffeine"]) ** 2
        dist += FEATURE_WEIGHTS["temp"] * (p_temp - prof["temp"]) ** 2

        # convert distance to a similarity logit (closer -> higher score)
        logit = -dist / 6.0

        # personality nudges
        logit += FLAVOR_BIAS.get(person["flavor"], {}).get(coffee, 0.0)
        logit += EXPERIENCE_BIAS.get(person["experience"], {}).get(coffee, 0.0)
        logit += TIME_BIAS.get(person["time_of_day"], {}).get(coffee, 0.0)
        if person["previous"] == coffee:
            logit += PREVIOUS_BIAS_STRENGTH

        scores[coffee] = logit

    coffees = list(scores.keys())
    logits = np.array([scores[c] for c in coffees])

    # temperature on the softmax controls how "noisy"/overlapping labels are.
    # < 1 sharpens the distribution around the best-matching coffee(s) while
    # still leaving room for overlap between adjacent profiles -- realistic
    # but learnable, rather than near-random.
    probs = softmax(logits / 0.55)
    return coffees, probs


def generate_dataset(n=N_SAMPLES):
    rows = []
    for _ in range(n):
        person = sample_person()
        coffees, probs = score_person_against_profiles(person)
        label = rng.choice(coffees, p=probs)
        row = dict(person)
        row["coffee_preference"] = label
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset(N_SAMPLES)

    out_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "coffee_preferences.csv")
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} rows -> {out_path}")
    print("\nLabel distribution:")
    print(df["coffee_preference"].value_counts())
