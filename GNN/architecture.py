from torch import nn
from torch_geometric.nn import SAGEConv, HeteroConv, GCNConv, GraphConv, GATConv
from torch_geometric.data import HeteroData
import torch
import pandas as pd
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader

class Encoder:
    """
    This class is responsible for creating a two way mapping of users and items to indices
    """
    def __init__(
            self,
            encoder=None
    ):
        """
        Args:
            encoder: Encoder type object 
        Remarks:
            all files are in JSON format
        """
        if encoder is None:
            self.user_id = {}
            self.id_user = {}
            self.item_id = {}
            self.id_item = {}
        else:
            self.user_id = encoder.user_id
            self.id_user = encoder.id_user
            self.item_id = encoder.item_id
            self.id_item = encoder.id_item
    
    def encode(
            self,
            df: pd.DataFrame
    ):
        """
        Args:
            df: pandas dataframe having two mandatory fields: userId, productId
        Raises:
            KeyError if userId and productId are missing
        """
        if 'userId' not in df.columns or 'productId' not in df.columns:
            raise KeyError('Fileds "userId" or "productId" are missing')
        try:
            max_userid = max(self.id_user.keys())
        except:
            max_userid = -1
        try:
            max_itemid = max(self.id_item.keys())
        except:
            max_itemid = -1
    
        for userid, productid in df[['userId', 'productId']].values:
            if userid not in self.user_id.keys():
                max_userid += 1
                self.user_id[userid] = max_userid
                self.id_user[max_userid] = userid
            if productid not in self.item_id.keys():
                max_itemid += 1
                self.item_id[productid] = max_itemid
                self.id_item[max_itemid] = productid

class GNN(nn.Module):
    def __init__(self, n_users: int, n_items: int, dim_item_feature: int, hidden_dim: int = 32):
        super().__init__()
        # Learnable embeddings
        self.user_emb = nn.Embedding(n_users, hidden_dim)
        self.item_emb = nn.Embedding(n_items, hidden_dim)

        self.item_feature_proj = nn.Linear(in_features=dim_item_feature, out_features=hidden_dim)

        self.conv1 = HeteroConv({
            ('user', 'rates', 'product') : SAGEConv(hidden_dim, hidden_dim),
            ('product', 'rated_by', 'user') : SAGEConv(hidden_dim, hidden_dim)
        })

        self.conv2 = HeteroConv({
            ('user', 'rates', 'product') : SAGEConv(hidden_dim, hidden_dim),
            ('product', 'rated_by', 'user') : SAGEConv(hidden_dim, hidden_dim)
        })
     
        self.projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
    
    def forward(self, input_data: HeteroData):
        """
        The input HeteroData should have 'user', 'product' keys and 
        ('user', 'rates', 'product') and ('product', 'rated_by', 'user') relations
        """
        data = HeteroData()
        data['user'].x = self.user_emb(input_data['user'].x)
        item_features = self.item_feature_proj(input_data['product_feature'].x)
        data['product'].x = self.item_emb(input_data['product'].x) + item_features
        # data['product'].x = item_features

        data['user', 'rates', 'product'].edge_index = input_data['user', 'rates', 'product'].edge_index
        data['product', 'rated_by', 'user'].edge_index = input_data['product', 'rated_by', 'user'].edge_index
        
        out_dict_1 = self.conv1(data.x_dict, data.edge_index_dict)
        out_dict_1['user'] = nn.functional.leaky_relu(out_dict_1['user'])
        out_dict_1['product'] = nn.functional.leaky_relu(out_dict_1['product'])

        out_dict_2 = self.conv2(out_dict_1, data.edge_index_dict)
        out_dict_2['user'] = nn.functional.leaky_relu(out_dict_2['user'])
        out_dict_2['product'] = nn.functional.leaky_relu(out_dict_2['product'])

        self.final_user_emb = torch.cat([data['user'].x, out_dict_1['user'], out_dict_2['user']], dim=1)
        self.final_item_emb = torch.cat([data['product'].x, out_dict_1['product'], out_dict_2['product']], dim=1)

    
    def predict(self, users: torch.Tensor, items: torch.Tensor):
        """
        Args:
            users: list of user nodes
            items: list of item nodes
            mu: average rating
        """
        users = self.final_user_emb[users]
        items = self.final_item_emb[items]
        ratings = torch.sum(users*items, dim=1)
        return ratings

class BPRLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, positive_ratings, negative_ratings):
        diff = positive_ratings - negative_ratings
        loss = nn.functional.sigmoid(diff)
        logloss = -torch.log(loss)
        return torch.mean(logloss)

class PairDatset(Dataset):
    def __init__(self, users, pos_items, neg_items):
        self.users = users
        self.pos_items = pos_items
        self.neg_items = neg_items
    
    def __len__(self):
        return self.users.shape[0]

    def __getitem__(self, id):
        return self.users[id], self.pos_items[id], self.neg_items[id]

class GNNTrainer:
    def __init__(
            self,
            train_data: pd.DataFrame,
            movie_features: pd.DataFrame,
            train_pair_data: np.ndarray,
            val_pair_data: np.ndarray
    ):
        """
        The 'train_data' and 'val_data' data should be a pandas dataframe having 'userId', 'productId' and 'rating' fields
        The 'movie_features' should have only one hot features only
        The 'train_pair_data' is numpy array of shape (N, 4), col (0,1) is +ve pair and (2,3) is -ve pair, same for 'val_pair_data'
        """
        # Encoding the data first
        self.enc = Encoder()
        self.enc.encode(train_data)
        self.save_encoder()
        encoded_train_data = train_data.copy()

        encoded_train_data['userId'] = encoded_train_data['userId'].map(lambda x: self.enc.user_id[x])
        encoded_train_data['productId'] = encoded_train_data['productId'].map(lambda x: self.enc.item_id[x])

        # Making Heterogeneous Graph
        n_users = train_data['userId'].unique().shape[0]
        n_products = train_data['productId'].unique().shape[0]

        # Training Graph
        self.train_data = HeteroData()
        self.train_data['user'].x = torch.tensor(range(n_users))
        self.train_data['product'].x = torch.tensor(range(n_products))
        self.train_data['user', 'rates', 'product'].edge_index = torch.tensor(encoded_train_data[['userId', 'productId']].values.T, dtype=torch.long)
        self.train_data['product', 'rated_by', 'user'].edge_index = torch.tensor(encoded_train_data[['productId', 'userId']].values.T, dtype=torch.long)
        # self.train_data['user', 'rates', 'product'].edge_attr = torch.tensor(encoded_train_data['rating'].values, dtype=torch.float)
        # self.train_data['product', 'rated_by', 'user'].edge_attr = torch.tensor(encoded_train_data['rating'].values, dtype=torch.float)


        # Adding product features
        selected_movie_features = movie_features[movie_features.index.isin(train_data['productId'].unique())]
        selected_movie_features.index = selected_movie_features.index.map(lambda x: self.enc.item_id[x])
        selected_movie_features.sort_index(inplace=True)
        self.train_data['product_feature'].x = torch.tensor(selected_movie_features.values, dtype=torch.float32)

        # prepare train-pair data
        users = torch.tensor(list(map(lambda x: self.enc.user_id[x], train_pair_data[:,0])))
        pos_items = torch.tensor(list(map(lambda x: self.enc.item_id[x], train_pair_data[:,1])))
        neg_items = torch.tensor(list(map(lambda x: self.enc.item_id[x], train_pair_data[:,3])))
        pairdata = PairDatset(users, pos_items, neg_items)
        self.train_pairloader = DataLoader(pairdata, 60000, num_workers=30)

        # prepare validation-pair data
        users = torch.tensor(list(map(lambda x: self.enc.user_id[x], val_pair_data[:,0])))
        pos_items = torch.tensor(list(map(lambda x: self.enc.item_id[x], val_pair_data[:,1])))
        neg_items = torch.tensor(list(map(lambda x: self.enc.item_id[x], val_pair_data[:,3])))
        pairdata = PairDatset(users, pos_items, neg_items)
        self.val_pairloader = DataLoader(pairdata, 60000, num_workers=30)


        # initialising the model
        self.model = GNN(
            n_users=n_users,
            n_items=n_products,
            dim_item_feature=self.train_data['product_feature'].x.shape[1]
        )

    def train(
            self,
            epochs: int = 10,
            lr: float = 10**(-3),
            decay: float = 10**(-4)
    ):
        torch.manual_seed(42)
        optimiser = torch.optim.Adam(self.model.parameters(), weight_decay=decay, lr=lr)
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        self.train_data = self.train_data.to(device) # type: ignore
        loss_fn = BPRLoss()

        self.train_losses = []
        self.val_losses = []

        for epoch in range(epochs):
            train_bpr_loss = []
            val_bpr_loss = []
            for j, (users, pos_items, neg_items) in enumerate(self.train_pairloader):
                self.model(self.train_data)
                users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
                pos_ratings = self.model.predict(users, pos_items)
                neg_ratings = self.model.predict(users, neg_items)
                loss = loss_fn(pos_ratings, neg_ratings)
            
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                train_bpr_loss.append(loss.item())

                if (j+1) % 20 == 0:
                    print(f"Epoch {epoch+1}, step {j+1} | BPR Loss - {loss.item()}")

            with torch.no_grad():
                # Validation loss
                for users, pos_items, neg_items in self.val_pairloader:
                    users, pos_items, neg_items = users.to(device), pos_items.to(device), neg_items.to(device)
                    pos_ratings = self.model.predict(users, pos_items)
                    neg_ratings = self.model.predict(users, neg_items)
                    val_loss = loss_fn(pos_ratings, neg_ratings)
                    val_bpr_loss.append(val_loss.item())

            self.train_losses.append(np.mean(train_bpr_loss))
            self.val_losses.append(np.mean(val_bpr_loss))
            print(f"At end of epoch {epoch+1}: train BPR loss = {self.train_losses[-1].item()} | validation BPR loss = {self.val_losses[-1].item()}")
            self.save_model(epoch)
        
        self.saveplot()



    def save_encoder(self):
        encpath = Path.cwd()/f'models/GNN/encoder.pkl'
        with open(encpath, "wb") as f:
            pickle.dump(self.enc, f)
            f.close()
    
    def save_model(self, epoch: int):
        basepath = Path.cwd()/f'models/GNN'
        # if (epoch+1) % 10 == 0:
        modelpath = basepath/f'model_{epoch+1}.pt'
        with open(modelpath, "wb") as f:
            torch.save(self.model, f)
            f.close()
        if self.val_losses[-1] == min(self.val_losses):
            modelpath = basepath/'best.pt'
            with open(modelpath, "wb") as f:
                torch.save(self.model, f)
                f.close()
    def saveplot(self):
        plt.plot(self.val_losses, label='val-BPR Loss')
        plt.plot(self.train_losses, label='train-BPR Loss')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('MSE Loss')
        plt.title(f'Train & Val loss')
        plt.savefig(Path.cwd()/f'models/GNN/result.png')
        plt.show()