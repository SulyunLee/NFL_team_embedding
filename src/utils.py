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

        return self.early_stop

