from reccomender import Reccomender
import numpy as np

rec = Reccomender("../data/combined.csv", 0.4, 0.98, 4)


def generate_reward(filename):
    if filename.startswith("5"):
        return np.random.choice([-20, 20], p=[0.9, 0.1])
    elif filename.startswith("3"):
        return np.random.choice([-20, 20], p=[0.4, 0.6])
    elif filename.startswith("10"):
        return np.random.choice([-20, 20], p=[0.4, 0.6])
    else:
        return 0


preference_counts = {"atheism": 0, "graphics": 0, "baseball": 0}
for _ in range(1000):
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

rec.visualize_embedding_map()
