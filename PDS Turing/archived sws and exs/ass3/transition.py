import numpy as np
class TransitionMatrix:
    def __init__(self, tr_mat):
        greater_zero = (0 >= tr_mat).all() 
        less_one = (tr_mat <= 1).all()
        self.probabilities = tr_mat
        self.tr_mat = tr_mat
        if type(tr_mat) is not np.ndarray:
            raise TypeError
        elif (greater_zero == False) and (less_one == False):
            raise ValueError
        
    def step(self):
        tr_mat = self.tr_mat
        tm_step = TransitionMatrix(tr_mat)
        tm_step.probabilities = self.probabilities * self.tr_mat
        return tm_step
