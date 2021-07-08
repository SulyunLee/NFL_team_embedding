import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, loss, model):
        score = loss
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score:
            self.counter += 1
            print("Early stopping counter: {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.counter, self.early_stop

def normalize(train, valid, test):
    '''
    This function normalizes input train, validation, and test datasets before feeding
    into a deep learning model.
    Uses mean zero normalization function.
    '''

    combined = np.zeros((train.shape[0] + valid.shape[0] + test.shape[0], train.shape[1]))
    combined[:train.shape[0]] = train
    combined[train.shape[0]:train.shape[0]+valid.shape[0],:] = valid
    combined[train.shape[0]+valid.shape[0]:,:] = test

    means = combined.mean(axis=0)
    stds = combined.std(axis=0)

    normalized_train = (train - means) / stds
    normalized_valid = (valid - means) / stds
    normalized_test = (test - means) / stds

    return torch.Tensor(normalized_train), torch.Tensor(normalized_valid), torch.Tensor(normalized_test)
    
