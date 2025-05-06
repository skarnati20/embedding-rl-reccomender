from reccomender import Reccomender
import numpy as np

rec = Reccomender("../data/combined.csv")

def generate_reward(filename):
    if filename.startswith("5"):
        return np.random.choice([0, 10], p=[0.9, 0.1])
    elif filename.startswith("3"):
        return np.random.choice([0, 10], p=[0.9, 0.1])
    elif filename.startswith("10"):
        return np.random.choice([0, 10], p=[0.1, 0.9])
    else:
        return 0

# Preference phase
preference_counts = {"atheism": 0, "graphics": 0, "baseball": 0}
for _ in range(500):
    file, embedding = rec.recommend_article()
    if file.startswith("5"):
        preference_counts["atheism"] += 1
    elif file.startswith("3"):
        preference_counts["graphics"] += 1
    elif file.startswith("10"):
        preference_counts["baseball"] += 1
    print(preference_counts)
    reward = generate_reward(file)
    rec.register_reaction(embedding, reward)

counts = {"atheism": 0, "graphics": 0, "baseball": 0}
for _ in range(50):
    file, _ = rec.recommend_article()
    if file.startswith("5"):
        counts["atheism"] += 1
    elif file.startswith("3"):
        counts["graphics"] += 1
    elif file.startswith("10"):
        counts["baseball"] += 1

print(counts)