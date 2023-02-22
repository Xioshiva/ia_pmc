from matplotlib import colors
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve
from sklearn.utils.extmath import row_norms
import PMC 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

scores_train = []
label = []
scores_std = []
scores_means = []
n_neuronnes = {5,10,20}
n_iterations = {500,1000,2000}

def pmc():
    print("pmc:")
    for n_neuronne in n_neuronnes:
        for n_iteration in n_iterations:
            legend = "for n_iteration: " + str(n_iteration) + " n_neuronne: " + str(n_neuronne)
            print(legend)
            clf = MLPClassifier(solver='sgd', activation='logistic', max_iter=n_iteration, hidden_layer_sizes=(n_neuronne,), learning_rate_init=0.1)
            clf.fit(data_train, class_train)
            print("\tpercentage accuracy of predict from clf.score (1 test): "+str(clf.score(data_test, class_test, sample_weight=None)))
            scores_train.append(clf.score(data_test, class_test, sample_weight=None))
            label.append("n:"+str(n_neuronne)+" i:"+str(n_iteration))
            plt.plot(clf.loss_curve_, label="i:"+str(n_iteration) + " n:"+ str(n_neuronne))
            scores = cross_val_score(clf, data_test, class_test, cv=10)
            print("\tmean accuracy %f with a standard deviation of %f from cross_val_score" % (scores.mean(), scores.std()))
            scores_std.append(str(scores.std()))  
            scores_means.append(str(scores.mean()))              
                
def random_forest(): 
    print("random forest:")
    clf = RandomForestClassifier(n_estimators=1000,criterion="gini",max_features="sqrt")
    clf.fit(data_train, class_train)
    print("\tpercentage accuracy of predict from clf.score (1 test): "+str(clf.score(data_test, class_test, sample_weight=None)))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(clf, data_test, class_test, ax=ax, alpha=0.8)
    scores = cross_val_score(clf, data_test, class_test, cv=10)
    print("\tmean accuracy %f with a standard deviation of %f from cross_val_score" % (scores.mean(), scores.std()))                
            
    

if __name__ == "__main__": 
    data_train, class_train = PMC.open_file_heart("./ressource/DATA/heart.dat") 
    data_train, data_test, class_train, class_test = train_test_split(data_train, class_train, test_size=0.1, random_state=0)
    pmc()  
    plt.title("Loss Curve by Iteration and nNeuronne", fontsize=14)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.grid()
    plt.show()
    
    data = []
    data.append(scores_train)
    data.append(scores_means)
    data.append(scores_std)
    test = np.transpose(data)
    
    rows_label=['scores train', 'scores means', 'scores std']
    df = pd.DataFrame(test, label, rows_label)
    print(df)
    
    random_forest()
    plt.show()