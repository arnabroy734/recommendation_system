# Collaborative Filtering Movie Recommendation System

## Problem Statement

This project implements a collaborative filtering based recommendation system that predicts user preferences for movies by analyzing patterns in user-item interactions. The system learns from historical user ratings to suggest relevant movies, addressing the challenge of information overload in movie selection for users.

## Dataset

The project utilizes a curated subset of the MovieLens Dataset (2022-2023), containing 17,888 unique users and 48,546 unique movies with 1,515,971 user reviews. This rich dataset provides comprehensive user-movie interaction patterns essential for training effective collaborative filtering models and evaluating recommendation performance.

## Methods

### 1. Matrix Factorization with Alternating Least Squares (ALS)

Implemented a collaborative filtering approach based on matrix factorization using the Alternating Least Squares method, inspired by the Netflix Prize paper ["Matrix Factorization Techniques for Recommender Systems"](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf) by Koren et al. The ALS algorithm was implemented from scratch, decomposing the user-item interaction matrix into lower-dimensional user and item feature matrices. This method alternately optimizes user and item factors while keeping the other fixed, effectively handling the sparse nature of rating data and providing scalable collaborative filtering.

### 2. Neural Graph Collaborative Filtering (NGCF)

Implemented the Neural Graph Collaborative Filtering approach based on the paper ["Neural Graph Collaborative Filtering"](https://arxiv.org/abs/1905.08108) by Wang et al. This method constructs a heterogeneous bipartite graph representing user-item interactions and employs graph neural networks to capture high-order connectivity. The implementation features two message passing layers using SageConv (GraphSAGE Convolution) to aggregate neighborhood information and learn enhanced user and item embeddings. The model is optimized using Bayesian Personalized Ranking (BPR) loss, which is specifically designed for implicit feedback recommendation tasks.

## Results

| Model | NDCG (Overall) | NDCG@20 |
|-------|----------------|---------|
| ALS   | 27.5%          | 2.98%   |
| GNN   | 35.5%          | 8.1%    |

The Neural Graph Collaborative Filtering model significantly outperforms the traditional ALS approach, achieving a 29% improvement in overall NDCG and a 172% improvement in NDCG@20, demonstrating the effectiveness of graph-based methods in capturing complex user-item relationships.

## Key Features

- **From-scratch ALS implementation** for matrix factorization
- **Graph Neural Network architecture** with heterogeneous graph construction
- **Comprehensive evaluation** using NDCG metrics
- **Scalable design** handling large-scale user-item interactions
- **BPR loss optimization** for implicit feedback learning


## References

1. Koren, Y., Bell, R., & Volinsky, C. "Matrix Factorization Techniques for Recommender Systems"
2. Wang, X., He, X., Wang, M., Feng, F., & Chua, T. S. (2019). "Neural Graph Collaborative Filtering". arXiv:1905.08108