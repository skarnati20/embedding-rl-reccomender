# Embedding Recommender System

An implementation of a recommender system based on string embeddings and a multi-armed bandit RL model.

## Overview

Recommender systems attempt to understand users' preferences. They may do so by using the patterns of similar users (Collaborative Filtering) or using signficant user data to predict the user's next desire, which underlies many Deep Learning approaches to recommender systems.

With the prominence of robust text embedding models, I wanted to understand how they could be used to map user preferences. The approach taken in this repository is based on the idea that we can map a user's interest as a collection of embeddings with attached preferences, making it readily applicable to a multi-armbed bandits recommendation system.

When recommending an article we would then perform a KNN of the available, unread articles based on one of the embeddings in the user's interest profile, sourced by choosing it from a softmax distribution over the embeddings, and updated through a gradient bandits approach.

## How it Works

WIP

## Structure

* `data/` - Includes relevant scripts for generating csv files. Locally, I store a copy of the 20 newsgroup dataset as a csv file here.
* `src/` - Includes a class for the recommender and the embedding distribution object.
* `test/` - Includes the benchmark python script.

## Benchmarking

In order to run the benchmarks you need to run the following command from the project directory.

```bash
python -m test.benchmark
```

Make sure you have a correctly formatted csv file and change the path as neccesary in `benchmark.py`. Your csv file must have a column for "filename", "embedding", and "category".

The benchmark assumes you are testing on the 20 newsgroup dataset and its categories.