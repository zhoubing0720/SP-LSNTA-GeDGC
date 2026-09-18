import numpy as np
import torch

class EarlyStopping:
    def __init__(self, patience=10, delta=1e-3):
        self.patience = patience
        self.counter = 0
        self.last_loss = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, loss):
        if self.last_loss is None:
            self.last_loss = loss
        # elif np.abs(loss - self.last_loss) / self.last_loss < self.delta :
        #     self.counter += 1
        #     self.last_loss = loss
        #     print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        #     if self.counter >= self.patience:
        #         self.early_stop = True
        elif loss > self.last_loss - self.delta:
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.last_loss = loss
            self.counter = 0
