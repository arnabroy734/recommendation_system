from MF_ALS.architecture import Encoder, ALSModelTrainer
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, shared_memory

train_df = pd.read_csv("data/movielens/train.csv", index_col=0)
small_df = train_df.iloc[0:1000]
print(small_df['userId'].unique().shape)
print(small_df['productId'].unique().shape)

trainer = ALSModelTrainer(regulariser=0.2, train_data=small_df)

trainer.train()

# diff_norm = np.linalg.norm(trainer.model.user_mat - trainer.updated_user_mat, axis=1)
# print(diff_norm)