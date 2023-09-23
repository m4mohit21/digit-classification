from sklearn.model_selection import train_test_split
# Standard scientific Python imports
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm

def find_acc(model, X_test, y_test):
    predicted = model.predict(X_test)
    return metrics.accuracy_score(y_test,predicted)

def tune_hparams(model,X_train, X_test, X_dev , y_train, y_test, y_dev,list_of_param_combination):
    best_acc = -1
    for param_group in list_of_param_combination:
        temp_model = model(**param_group)
        temp_model.fit(X_train,y_train)
        acc = find_acc(temp_model,X_dev,y_dev)
        if acc > best_acc:
            best_acc = acc
            best_model = temp_model
            optimal_param = param_group
    train_acc= find_acc(best_model,X_train,y_train) 
    dev_acc = find_acc(best_model,X_dev,y_dev)
    test_acc =  find_acc(best_model,X_test,y_test)
    return train_acc, dev_acc, test_acc, optimal_param
    


def get_combinations(param,values,combinations):    
    new_combinations = []
    for value in values:
        for combination in combinations:
            combination[param] = value
            new_combinations.append(combination.copy())    
    return new_combinations

def get_hyperparameter_combinations(dict_of_param_lists):    
    base_combinations = [{}]
    for param_name, param_values in dict_of_param_lists.items():
        base_combinations = get_combinations(param_name, param_values, base_combinations)
    return base_combinations