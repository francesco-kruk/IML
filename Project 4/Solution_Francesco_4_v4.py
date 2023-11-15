# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


torch.manual_seed(0)
random.seed(0)

def load_data():
    """
    This function loads the data from the csv files and returns it as numpy arrays.

    input: None
    
    output: x_pretrain: np.ndarray, the features of the pretraining set
            y_pretrain: np.ndarray, the labels of the pretraining set
            x_train: np.ndarray, the features of the training set
            y_train: np.ndarray, the labels of the training set
            x_test: np.ndarray, the features of the test set
    """
    x_pretrain = pd.read_csv("pretrain_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_pretrain = pd.read_csv("pretrain_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_train = pd.read_csv("train_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1).to_numpy()
    y_train = pd.read_csv("train_labels.csv.zip", index_col="Id", compression='zip').to_numpy().squeeze(-1)
    x_test = pd.read_csv("test_features.csv.zip", index_col="Id", compression='zip').drop("smiles", axis=1)
    return x_pretrain, y_pretrain, x_train, y_train, x_tes

class Net(nn.Module):
    """
    The model class, which defines our feature extractor used in pretraining.
    """
    def __init__(self):
        """
        The constructor of the model.
        """
        super().__init__()
        # TODO: Define the architecture of the model. It should be able to be trained on pretraing data 
        # and then used to extract features from the training and test data.
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features=1000, out_features=1000, bias=True),
            torch.nn.BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.6, inplace=False),
            torch.nn.Linear(in_features=1000, out_features=800, bias=True),
            torch.nn.ReLU()

        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=800, out_features=1000, bias=True),
            torch.nn.Dropout(p=0.6, inplace=False),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(1000, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            torch.nn.Linear(in_features=1000, out_features=1000, bias=True)
        )

    def forward(self, x):
        """
        The forward pass of the model.

        input: x: torch.Tensor, the input to the model

        output: x: torch.Tensor, the output of the model
        """
        # TODO: Implement the forward pass of the model, in accordance with the architecture 
        # defined in the constructor.
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
def model_pretraining(x, y, batch_size=512, eval_size=2048):
    """
    This function trains the feature extractor on the pretraining data and returns a function which
    can be used to extract features from the training and test data.g
    input: x: np.ndarray, the features of the pretraining set
              y: np.ndarray, the labels of the pretraining set
                batch_size: int, the batch size used for training
                eval_size: int, the size of the validation set
            
    output: make_features: function, a function which can be used to extract features from the training and test data
    """
    # Pretraining data loading
    x_tr, x_val, y_tr, y_val = train_test_split(x, y, test_size=eval_size, random_state=0, shuffle=True)
    x_tr, x_val = torch.tensor(x_tr, dtype=torch.float), torch.tensor(x_val, dtype=torch.float)
    y_tr, y_val = torch.tensor(y_tr, dtype=torch.float), torch.tensor(y_val, dtype=torch.float)

    # model declaration
    model = Net()
    if torch.cuda.is_available():
        model = model.cuda()
    
    # TODO: Implement the training loop. The model should be trained on the pretraining data. Use validation set 
    # to monitor the loss.
    n_epochs = 100
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    training_loss = []
    validation_loss = []

    train_loader = torch.utils.data.DataLoader(x_tr, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(x_val, batch_size=batch_size, shuffle=True)

    for epoch in range(n_epochs):
        model.train()
        for data_train in train_loader:
            if torch.cuda.is_available():
                data_train = data_train.cuda()
            optimizer.zero_grad()
            target_train = model(data_train)
            loss = criterion(target_train, data_train)
            loss.backward()
            optimizer.step()
            train_loss = loss.item()
            training_loss.append(train_loss)

        model.eval()
        with torch.no_grad():
            for data_val in val_loader:
                if torch.cuda.is_available():
                    data_val = data_val.cuda()
                target_val = model(data_val)
                loss = criterion(target_val, data_val)
                val_loss = loss.item()
                validation_loss.append(val_loss)

        print(f"Epoch {epoch + 1}/{n_epochs}, Training Loss: {train_loss}, Validation Loss: {val_loss}")

    return model

def make_features(x, model):
    """
    This function extracts features from the training and test data, used in the actual pipeline
    after the pretraining.

    input: x: np.ndarray, the features of the training or test set

    output: features: np.ndarray, the features extracted from the training or test set, propagated
    further in the pipeline
    """
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    # TODO: Implement the feature extraction, a part of a pretrained model used later in the pipeline.
    with torch.no_grad():
        x = torch.tensor(x, dtype=torch.float)
        features = model(x)
    return features.cpu().numpy()


def make_pretraining_class(feature_extractors):
    """
    The wrapper function which makes pretraining API compatible with sklearn pipeline
    
    input: feature_extractors: dict, a dictionary of feature extractors

    output: PretrainedFeatures: class, a class which implements sklearn API
    """

    class PretrainedFeatures(BaseEstimator, TransformerMixin):
        """
        The wrapper class for Pretraining pipeline.
        """
        def __init__(self, *, feature_extractor=None, mode=None):
            self.feature_extractor = feature_extractor
            self.mode = mode

        def fit(self, X=None, y=None):
            return self

        def transform(self, X):
            assert self.feature_extractor is not None
            X_new = feature_extractors[self.feature_extractor](X)
            return X_new
        
    return PretrainedFeatures


def get_regression_model():
    """
    This function returns the regression model used in the pipeline.

    input: None

    output: model: sklearn compatible model, the regression model
    """
    # TODO: Implement the regression model. It should be able to be trained on the features extracted
    # by the feature extractor.
    # model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10])
    model = RandomForestRegressor(n_estimators=200)
    # kernel = RBF(length_scale_bounds=(1e-5, 1e5)) + WhiteKernel()
    # model = GaussianProcessRegressor(kernel=kernel)
    # model = XGBRegressor(n_estimators=10, objective='reg:squarederror', random_state=1, reg_lambda=0.001,
               # eval_metric="logloss", use_label_encoder=False, n_jobs=18, grow_policy="lossguide", max_leaves=0)
    return model

# Main function. You don't have to change this
if __name__ == '__main__':
    # Load data
    x_pretrain, y_pretrain, x_train, y_train, x_test = load_data()

    # Utilize pretraining data by creating feature extractor which extracts lumo energy 
    # features from available initial features
    pretrained_model = model_pretraining(x_pretrain, y_pretrain)
    x_train_new = make_features(x_train, pretrained_model)
    x_test_new = make_features(x_test.to_numpy(), pretrained_model)
    # PretrainedFeatureClass = make_pretraining_class({"pretrain": feature_extractor})
    print ("Features extracted!")
    
    # regression model
    regression_model = get_regression_model()
    print("regression model created!")

    y_pred = np.zeros(x_test.shape[0])
    # TODO: Implement the pipeline. It should contain feature extraction and regression. You can optionally
    # use other sklearn tools, such as StandardScaler, FunctionTransformer, etc.
    pipe = Pipeline([('preprocessing', StandardScaler()), ('regression', regression_model)])
    pipe.fit(x_train_new, y_train)
    y_pred = pipe.predict(x_test_new)
    print("Prediction created!")
    print("Score: ", regression_model.score(x_train_new, y_train))

    assert y_pred.shape == (x_test.shape[0],)
    y_pred = pd.DataFrame({"y": y_pred}, index=x_test.index)
    y_pred.to_csv("Results_Francesco_4_v4.csv", index_label="Id")
    print("Predictions saved!")