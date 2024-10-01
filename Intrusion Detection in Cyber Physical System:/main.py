from GWO import OriginalGWO
from HHO import OriginalHHO
from SHO import OriginalSHO
from HRFSG import *
from PSO import OriginalPSO
from classifer import *
import matplotlib
from Objective_fun import obj_fun
from plot_res import *
from load_save import save,load
matplotlib.use('TkAgg', force=True)
from datagen import datagen

def full_analysis():
    datagen()
    X_train = load('X_train')
    X_test = load('X_test')
    y_train = load('y_train')
    y_test = load('y_test')

    # Optimization parameters
    lb = (np.zeros([1, X_train[i].shape[1]]).astype('int16')).tolist()[0]
    ub = (np.ones([1, X_train[i].shape[1]]).astype('int16')).tolist()[0]

    prb_size = len(lb)
    pop_size = 10
    epoch = 50
    problem_dict1 = {
        "fit_func": obj_fun,
        "lb": lb,
        "ub": ub,
        "minmax": "min"
    }

    # Perform optimization
    pro_best_solution, best_value = HRFSG(obj_fun, lb, ub, prb_size, pop_size,epoch)
    soln = np.round(pro_best_solution)
    selected_indices = np.where(soln == 1)[0]

    # Converting lists to numpy arrays for advanced indexing
    X_train_np = np.array(X_train[i])
    X_test_np = np.array(X_test[i])
    y_train_np = np.array(y_train[i])
    y_test_np = np.array(y_test[i])

    selected_x_train = X_train_np[:, selected_indices]
    selected_x_test = X_test_np[:, selected_indices]
    selected_y_train = y_train_np
    selected_y_test = y_test_np


     #GWO
    epoch = 1000
    pop_size = 50
    model = OriginalGWO(epoch, pop_size)
    best_position, best_fitness = model.solve(problem_dict1)
    print(f"Solution: {best_position}, Fitness: {best_fitness}")


    #HHO
    model = OriginalHHO(epoch, pop_size)
    best_position, best_fitness = model.solve(problem_dict1)
    save('CSA_best_position',best_position)


    #SHO

    pop_size = 50
    h_factor = 5.0
    N_tried = 10
    model = OriginalSHO(epoch, pop_size, h_factor, N_tried)
    best_position, best_fitness = model.solve(problem_dict1)
    save('HBA_best_position',best_position)

    #PSO
    c1 = 2.05
    c2 = 2.05
    w_min = 0.4
    w_max = 0.9
    model = OriginalPSO(epoch, pop_size, c1, c2, w_min, w_max)
    best_position, best_fitness = model.solve(problem_dict1)
    save('PSO_best_position',best_position)


    # PROPOSED
    ypre, met = rbm_wdlstm(selected_x_train, selected_y_train, selected_x_test, selected_y_test)
    save('proposed_met', met)



    #CNN
    ypre, met =CNN(X_train, y_train, X_test, y_test)
    save('CNN_met', met)


    #DNN
    ypre, met = DNN(X_train, y_train, X_test, y_test)
    save('DNN_met', met)



    #RNN
    ypre, met = RNN(X_train, y_train, X_test, y_test)
    save('RNN_met', met)


an=0
if an==1:
    full_analysis()

plot_res()
plot_res2()



