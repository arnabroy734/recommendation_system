import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Pool, shared_memory
from pathlib import Path
import pickle
from functools import partial

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
            train_data: pd.DataFrame,
            validation_data: pd.DataFrame,
            dim: int,
            tol: float = 10**(-4)
    ):
        """
        The data should be a pandas dataframe having 'userId', 'productId' and 'rating' fields
        """
        # Encoding the data
        self.enc = Encoder()
        self.enc.encode(train_data)
        self.save_encoder()

        # Model Initialisation
        self.model = ALSModel (
            mu=train_data['rating'].mean(),
            n_users=len(self.enc.user_id.keys()),
            n_items=len(self.enc.item_id.keys()),
            dim=dim
        )
        self.train_data = train_data
        self.validation_data = validation_data[validation_data['productId'].isin(self.enc.item_id.keys())]
        self.validation_data = self.validation_data[self.validation_data['userId'].isin(self.enc.user_id.keys())]
        
        self.regulariser = regulariser
        self.tol = tol

        self.user_vector_norms = np.zeros(
            shape=(len(self.enc.user_id.keys()), )
        )
        self.item_vector_noms = np.zeros(
            shape=(len(self.enc.item_id.keys()), )
        )
        self.train_mse = []
        self.val_mse = []

    def _update_one_user(self, u):
        """
        This method updates one user vector and bias 
        """
        userId = self.enc.id_user[u]
        N = self.train_data.groupby(by='userId')['rating'].count()[userId] # total items user u rated
        user_df = self.train_data[self.train_data['userId'] == userId] # subset data for user u
        items = user_df['productId'].map(lambda x: self.enc.item_id[x]).values # items user u rated
        item_vectors = self.model.item_mat[items]
            
        denominator_matrix = item_vectors.T@item_vectors + self.regulariser*N*np.identity(self.model.dim)

        user_ratings = user_df['rating'].values.reshape((-1,1))
        user_bias = self.model.user_bias[u]
        item_bias = self.model.item_bias[items]
        mu = self.model.mu

        numerator_matrix = user_ratings*item_vectors - N*(mu + user_bias)*item_vectors - item_bias*item_vectors
        numerator_matrix = np.sum(numerator_matrix, axis=0, keepdims=True).reshape((-1,1))  
            
        updated_pu = np.linalg.pinv(denominator_matrix)@numerator_matrix # update user vector
        updated_pu = updated_pu.flatten()

        user_vector = self.model.user_mat[u].reshape((-1,1))
        updated_bias = np.sum(user_ratings.flatten()) - N*mu - np.sum(item_bias.flatten()) - np.sum((item_vectors@user_vector).flatten())
        updated_bias = updated_bias/(self.regulariser*N + 1)

        # open shared memory
        sham_user_mat = shared_memory.SharedMemory(name='user_mat')
        sham_user_bias = shared_memory.SharedMemory(name='user_bias')
        shared_user_mat = np.ndarray(
            shape=self.model.user_mat.shape,
            dtype=self.model.user_mat.dtype,
            buffer=sham_user_mat.buf
        )
        shared_user_bias = np.ndarray(
            shape=self.model.user_bias.shape,
            dtype=self.model.user_bias.dtype,
            buffer=sham_user_bias.buf
        )
        shared_user_mat[u] = updated_pu
        shared_user_bias[u] = updated_bias

        # print(f"User {sham_user_mat.name}, {u}, vector updated.")

        sham_user_mat.close()
        sham_user_bias.close()
        # sham_user_mat.unlink()
            # print(self.updated_user_mat[u])


    def _update_one_product(self, i):
        """
        This method updates one item matrix and bias 
        """
        productId = self.enc.id_item[i]
        M = self.train_data.groupby(by='productId')['rating'].count()[productId] # total items item i rated
        item_df = self.train_data[self.train_data['productId'] == productId] # subset data for item i
        users = item_df['userId'].map(lambda x: self.enc.user_id[x]).values # users items i rated by
        user_vectors = self.model.user_mat[users]
            
        denominator_matrix = user_vectors.T@user_vectors + self.regulariser*M*np.identity(self.model.dim)

        item_ratings = item_df['rating'].values.reshape((-1,1))
        item_bias = self.model.item_bias[i]
        user_bias = self.model.user_bias[users]
        mu = self.model.mu

        numerator_matrix = item_ratings*user_vectors - M*(mu + item_bias)*user_vectors - user_bias*user_vectors
        numerator_matrix = np.sum(numerator_matrix, axis=0, keepdims=True).reshape((-1,1))  
            
        updated_qi = np.linalg.pinv(denominator_matrix)@numerator_matrix # update user vector
        updated_qi = updated_qi.flatten()

        item_vector = self.model.item_mat[i].reshape((-1,1))
        updated_bias = np.sum(item_ratings.flatten()) - M*mu - np.sum(user_bias.flatten()) - np.sum((user_vectors@item_vector).flatten())
        updated_bias = updated_bias/(self.regulariser*M + 1)

        # open shared memory
        sham_item_mat = shared_memory.SharedMemory(name='item_mat')
        sham_item_bias = shared_memory.SharedMemory(name='item_bias')
        shared_item_mat = np.ndarray(
            shape=self.model.item_mat.shape,
            dtype=self.model.item_mat.dtype,
            buffer=sham_item_mat.buf
        )
        shared_item_bias = np.ndarray(
            shape=self.model.item_bias.shape,
            dtype=self.model.item_bias.dtype,
            buffer=sham_item_bias.buf
        )
        shared_item_mat[i] = updated_qi
        shared_item_bias[i] = updated_bias

        # print(f"User {sham_user_mat.name}, {u}, vector updated.")

        sham_item_mat.close()
        sham_item_bias.close()
        # sham_user_mat.unlink()
            # print(self.updated_user_mat[u])
    
    def train(self, T=3):
        """
        This method will train by ALS method
        Args:
            T(int) : maximum no. of epochs
        """
        sham_user_mat = shared_memory.SharedMemory(
            name='user_mat',
            create=True,
            size=self.model.user_mat.nbytes
        )
        sham_user_bias = shared_memory.SharedMemory(
            name='user_bias',
            create=True,
            size=self.model.user_bias.nbytes
        )
        sham_item_mat = shared_memory.SharedMemory(
            name='item_mat',
            create=True,
            size=self.model.item_mat.nbytes
        )
        sham_item_bias = shared_memory.SharedMemory(
            name='item_bias',
            create=True,
            size=self.model.item_bias.nbytes
        )
        for t in range(T):
            
            self.updated_user_mat = np.ndarray(
                shape=self.model.user_mat.shape,
                dtype=self.model.user_mat.dtype,
                buffer=sham_user_mat.buf
            )
            self.updated_user_bias = np.ndarray(
                shape=self.model.user_bias.shape,
                dtype=self.model.user_bias.dtype,
                buffer=sham_user_bias.buf
            )
            self.updated_user_mat[:] = self.model.user_mat[:]
            with Pool(processes=multiprocessing.cpu_count()) as pool:
                pool.map(self._update_one_user, [u for u in self.enc.id_user.keys()])
                diff_norm = np.linalg.norm(self.model.user_mat - self.updated_user_mat, axis=1)
                diff_bias = np.abs(self.model.user_bias - self.updated_user_bias)

                print(f"At epoch - {t}: max user diff norm: {np.max(diff_norm)} | max user bias diff : {np.max(diff_bias)}")
                self.model.user_mat[:] = self.updated_user_mat[:]
                self.model.user_bias[:] = self.updated_user_bias[:]

            
            self.updated_item_mat = np.ndarray(
                shape=self.model.item_mat.shape,
                dtype=self.model.item_mat.dtype,
                buffer=sham_item_mat.buf
            )
            self.updated_item_bias = np.ndarray(
                shape=self.model.item_bias.shape,
                dtype=self.model.item_bias.dtype,
                buffer=sham_item_bias.buf
            )
            self.updated_item_mat[:] = self.model.item_mat[:]
            with Pool(processes=multiprocessing.cpu_count()) as pool:
                pool.map(self._update_one_product, [i for i in self.enc.id_item.keys()])
                diff_norm = np.linalg.norm(self.model.item_mat - self.updated_item_mat, axis=1)
                diff_bias = np.abs(self.model.item_bias - self.updated_item_bias)

                print(f"At epoch - {t}: max item diff norm: {np.max(diff_norm)} | max item bias diff : {np.max(diff_bias)}")
                self.model.item_mat[:] = self.updated_item_mat[:]
                self.model.item_bias[:] = self.updated_item_bias[:]
            
            # Run validation
            self.batch_predict(self.validation_data)
            self.batch_predict(self.train_data)
            val_mse = self.validation_data['rating'] - self.validation_data['prediction']     
            val_mse = np.mean(np.square(val_mse.values)) #type: ignore
            train_mse = self.train_data['rating'] - self.train_data['prediction']     
            train_mse = np.mean(np.square(train_mse.values))# type: ignore
            print(f"At end of epoch {t}| validation error: {val_mse} | train error: {train_mse}\n")

            self.train_mse.append(train_mse)
            self.val_mse.append(val_mse)
            
            # Save best model
            self.save_model()
                

        sham_user_mat.close()
        sham_user_mat.unlink()
        sham_item_mat.close()
        sham_item_mat.unlink()
    
    def batch_predict(self, data: pd.DataFrame, upper_cap=5.0, lower_cap=0.5):
        def predict(userId, productId):
            user = self.enc.user_id[userId],
            item = self.enc.item_id[productId]
            rating = self.model.predict_rating (user, item)[0] #type: ignore
            rating = min(rating, upper_cap)
            rating = max(rating, lower_cap)
            return rating
        
        data['prediction'] = data.apply(lambda x: predict(x['userId'], x['productId']), axis=1)
    
    def save_encoder(self):
        encpath = Path.cwd()/f'models/MF_ALS/encoder.pkl'
        with open(encpath, "wb") as f:
            pickle.dump(self.enc, f)
            f.close()

    def save_model(self):
        bestpath = Path.cwd()/f'models/MF_ALS/best.pkl'
        if np.max(self.val_mse) == self.val_mse[-1]:
            with open(bestpath, "wb") as f:
                pickle.dump(self.model, f)
                f.close()
    
class ALSInference:
    def __init__(self):
        """
        Load the best model and encoder from pre-defined path
        """ 
        modelpath = Path.cwd()/'models/MF_ALS/best.pkl'
        encpath = Path.cwd()/'models/MF_ALS/encoder.pkl'
        with open(modelpath, "rb") as f:
            self.model = pickle.load(f)
            f.close()
        with open(encpath, "rb") as f:
            self.enc = pickle.load(f)
            f.close()
        self.products = np.array(list(self.enc.id_item.keys()))

    def recommend(self, userId, top_k=5):
        """
        Recommend top_k products to the the userId
        """
        user = self.enc.user_id[userId]
        predictions = np.apply_along_axis(
            func1d=partial(self.model.predict_rating, user),
            arr=self.products,
            axis=0
        )
