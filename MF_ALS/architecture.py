import pandas as pd
import numpy as np

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

class ALSModel:
    """
    This is an abstract representation of matrix factorisation model.
    """
    def __init__(
            self,
            mu: float,
            n_users: int,
            n_items: int,
            dim: int = 128
    ):
        """
        Args:
            mu(float): mean rating for all user-item pairs
            n_users(int): no. of users
            n_items(int): no. of items
            dim(int): latent dimension of user and item vector
        """
        self.mu = mu        
        self.dim = dim
        self.user_mat = np.random.random(size=(n_users, dim))
        self.item_mat = np.random.random(size=(n_items, dim))
        self.user_bias = np.random.random(size=(n_users, 1))
        self.item_bias = np.random.random(size=(n_items, 1))
    
    def predict_rating(self, userid : int, itemid : int):
        """
        Predict rating of a user-item interaction
        """
        return self.mu + self.user_bias[userid] + self.item_bias[itemid] + self.user_mat[userid]@self.item_mat[itemid]

class ALSModelTrainer:
    """
    This class is designed for training by ALS method
    """
    def __init__(
            self,
            regulariser: float,
            data: pd.DataFrame,
            tol: float = 10**(-4)
    ):
        """
        The data should be a pandas dataframe having 'userId', 'productId' and 'rating' fields
        """
        # Encoding the data
        self.enc = Encoder()
        self.enc.encode(data)

        # Model Initialisation
        self.model = ALSModel (
            mu=data['rating'].mean(),
            n_users=len(self.enc.user_id.keys()),
            n_items=len(self.enc.item_id.keys()),
            dim=128
        )
        self.regulariser = regulariser
        self.tol = tol
    
    def train(self):
        """
        This method will train
        """
        pass

    