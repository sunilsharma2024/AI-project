import numpy as np
from classifer import rbm_wdlstm
from load_save import *



def obj_fun(soln):
    X_train = load('X_train')
    X_test = load('X_test')
    y_train = load('y_train')
    y_test = load('y_test')

    # Feature selection
    soln = np.round(soln)
    pred, met= rbm_wdlstm(X_train, y_train, X_test, y_test,soln)
    fit = 1 / met[0]

    return fit
