from tensorflow import keras
import numpy as np
import tempfile
from .par_base import BaseModel
from .signals import ContinuousSignal,CategoricalSignal
from sklearn import metrics


class Autoencoder(BaseModel):
    '''
    The vanilla version of autoencoder for anomaly detection
    
    
    Parameters
    ----------
    signals : list
        the list of signals the model is dealing with.
        Signals that are both input and output will be included.
    '''
    def __init__(self, signals):
        self.signals = signals
        self.estimator = None
        self.feats = []
        for signal in self.signals:
            if isinstance(signal, ContinuousSignal):
                self.feats.append(signal.name)
            if isinstance(signal, CategoricalSignal):
                self.feats.extend(signal.get_onehot_feature_names())
 
    def score_samples(self, x): 
        """
        get the anomaly scores for the input samples
        
        Parameters
        ----------
        data : ndarray
            the numpy array from where the samples are extracted
        return_evidence : bool, default is False
            whether to return observation and reconstruction for evidence
        
        Returns
        -------
        ndarray 
            the anomaly scores, matrix of shape = [n_samples,]
        """
        y_pred = self.estimator.predict(x)
        anomaly_scores = []
        for i in range(len(x)):
            err = metrics.mean_absolute_error(x[i],y_pred[i])
            anomaly_scores.append(err)
        return np.array(anomaly_scores)
    
 
    def train(self, train_x, hidden_dim, num_hidden_layers=1, 
                optimizer='adam',batch_size=64,epochs=10,verbose=0):
        """
        Build and train a vanilla version of autoencoder
        
        Parameters
        ----------
        train_data : ndarray or list of ndarray
            the numpy array from where the training samples are extracted
        val_data : ndarray or list of ndarray
            the numpy array from where the validation samples are extracted
        hidden_dim : int
            the latent dimension of encoder and decoder layers
        num_hidden_layers : int, default is 1
            the number of hidden LSTM layers
        optimizer : string or optimizer, default is 'adam'
            the optimizer for gradient descent
        batch_size : int, default is 256
            the batch size
        epochs : int, default is 10
            the maximum epochs to train the model
        save_best_only : int, default is True
            whether to save the model with best validation performance during training
        verbose : int, default is 0
            0 indicates silent, higher values indicate more messages will be printed
        
        Returns
        -------
        Autoencoder
            self
        
        """
        
        keras.backend.clear_session()
        
        input_dim = len(self.feats)
        self.estimator = self._make_network(input_dim, hidden_dim, num_hidden_layers)
        self.estimator.compile(optimizer=optimizer,loss='mse')
        
        checkpoint_path = tempfile.gettempdir()+'/Autoencoder.ckpt'
        cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,save_best_only=True, save_weights_only=True)                          
        self.estimator.fit(train_x,train_x, epochs=epochs, validation_split = 0.2, callbacks=[cp_callback], batch_size=batch_size, verbose=verbose)
        self.estimator.load_weights(checkpoint_path)
        return self

    def predict(self, x):
        """
        Predict the reconstructed samples
        
        Parameters
        ----------
        data : ndarray
            the numpy array from where the samples are extracted
            
        Returns
        -------
        ndarray 
            the reconstructed outputs, matrix of shape = [n_samples, n_feats] 
        """
        if self.estimator is None:
            return None
        y_hat = self.estimator.predict(x)
        y_pred = np.reshape(y_hat,(-1,len(self.feats)))
        return y_pred
    
    def score(self,x):
        """
        Calculate the negative MAE score on the given data. 
        
        Parameters
        ----------
        data : ndarray
            the numpy array from where the samples are extracted
            
        Returns
        -------
        float
            the negative mae score
        """
        y_hat = self.estimator.predict(x)
        score = np.abs(x-y_hat).mean()
        return -score
    
    def save_model(self,model_path=None, model_id=None):
        """
        save the model to files
        
        Parameters
        ----------
        model_path : string, default is None
            the target folder whether the model files are saved.
            If None, a tempt folder is created
        model_id : string, default is None
            the id of the model, must be specified when model_path is not None
            
        """
        
        model_path = super().save_model(model_path, model_id) 
        self.estimator.save(model_path+'/Autoencoder.h5')
    
    def load_model(self,model_path=None, model_id=None):
        """
        load the model from files
        
        Parameters
        ----------
        model_path : string, default is None
            the target folder whether the model files are located
            If None, load models from the tempt folder
        model_id : string, default is None
            the id of the model, must be specified when model_path is not None
            
        Returns
        -------
        Autoencoder
            self
        """
        model_path = super().load_model(model_path, model_id) 
        self.estimator = keras.models.load_model(model_path+'/Autoencoder.h5')
        return self
    
     
    def _make_network(self, input_dim, hidden_dim, num_hidden_layers):
        x_t = keras.Input(shape=(input_dim),name='x_t')
         
        interval = (input_dim-hidden_dim)//(num_hidden_layers+1)
         
        hidden_dims = []
        hid_dim = max(1,input_dim-interval)
        hidden_dims.append(hid_dim)
        f_dense1 = keras.layers.Dense(hid_dim, activation='relu',name='g_dense1')(x_t)
        for i in range(1,num_hidden_layers):
            hid_dim = max(1,input_dim-interval*(i+1))
            if i == 1:
                f_dense = keras.layers.Dense(hid_dim, activation='relu') (f_dense1)
            else:
                f_dense = keras.layers.Dense(hid_dim, activation='relu') (f_dense)
            hidden_dims.append(hid_dim)
        if num_hidden_layers > 1:
            z_layer = keras.layers.Dense(hidden_dim,name='z_layer')(f_dense)
        else:
            z_layer = keras.layers.Dense(hidden_dim,name='z_layer')(f_dense1)
        
        
        g_dense1 = keras.layers.Dense(hidden_dims[len(hidden_dims)-1], activation='relu',name='h_dense1')(z_layer)
        for i in range(1,num_hidden_layers):
            if i == 1:
                g_dense = keras.layers.Dense(hidden_dims[len(hidden_dims)-1-i], activation='relu') (g_dense1)
            else:
                g_dense = keras.layers.Dense(hidden_dims[len(hidden_dims)-1-i], activation='relu') (g_dense)
        
        if num_hidden_layers > 1:
            g_out = keras.layers.Dense(input_dim, activation='linear',name='dec_out') (g_dense)
        else:
            g_out = keras.layers.Dense(input_dim, activation='linear',name='dec_out') (g_dense1)
        model = keras.Model(x_t,g_out,name='ae')
        return model