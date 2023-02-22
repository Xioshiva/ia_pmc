from os import write
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import json

def open_file_heart(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    data = []
    classes = []
    lines = lines[18:]
    for line in lines:
        line = line.rstrip("\n")
        data.append(line.split(','))
        data[-1] = [float(i) for i in data[-1]]
        classes.append(int(data[-1].pop(len(data[-1])-1))-1)
    data = np.asarray(data,dtype=np.float32)
    scaler = preprocessing.MinMaxScaler().fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled, classes

def open_file_bands(filename):
    f = open(filename, 'r')
    lines = f.readlines()
    data = []
    classes = []
    lines = lines[24:]
    for line in lines:
        line = line.rstrip("\n")
        data.append(line.split(','))
        bands_class = data[-1].pop(len(data[-1])-1)
        classes.append(int(bands_class == "band"))
        data[-1] = [float(i) for i in data[-1]]
    data = np.asarray(data,dtype=np.float32)
    scaler = preprocessing.MinMaxScaler().fit(data)
    data_scaled = scaler.transform(data)
    return data_scaled, classes

def sig(x):
    return 1.0/(1.0 + np.exp(-x))

def make_prediction_sig(row, weights):
    activation = weights[-1]
    for i in range(0, len(row)-1):
        activation += weights[i] * row[i]
    return sig(activation)

def make_prediction_sig_bool(row, weights):
    activation = make_prediction_sig(row, weights)
    if activation >= 0.5:
        return 1
    return 0

def train_weights_multi(data, expected_class, nbr_neurons, rate, epoch):
    hiden_neurons = np.random.uniform(-0.5, 0.5, size=(nbr_neurons, len(data[0]) + 1))
    output_weights = np.random.uniform(-0.5, 0.5, size=(nbr_neurons + 1))
    for iteration in range(epoch):
        for i,row in enumerate(data):
            predictions = []
            for neuron in hiden_neurons:
                predictions.append(make_prediction_sig(row, neuron))
            out_pred = make_prediction_sig(predictions,output_weights)
            delta = out_pred*(1 - out_pred)*(expected_class[i] - out_pred)
            output_weights[-1] += rate * delta
            for j in range(len(output_weights)-1):
                output_weights[j] += rate * delta * predictions[j] 
    return hiden_neurons, output_weights


def bagging(data, classes, nbr_of_pmc):
    list_hiden_neurons, list_output_weights = ([] for i in range(2))
    nbr_neurons = 100
    epoch = 1000
    rate = 0.1
    data, data_test, classes, class_test = train_test_split(data, classes, test_size=0.1, random_state=0)
    for i in range(nbr_of_pmc):
        data_train, _, class_train, _ = train_test_split(data, classes, test_size=0.5, random_state=0)
        hiden_neurons,output_weights = train_weights_multi(data_train, class_train, nbr_neurons, rate,epoch)
        list_hiden_neurons.append(hiden_neurons)
        list_output_weights.append(output_weights)
        print(i)
    sum_error = 0
    for i, row in enumerate(data_test):
        list_out_pred = []
        for j in range(nbr_of_pmc):
            predictions = []
            for neuron in list_hiden_neurons[j]:
                predictions.append(make_prediction_sig(row, neuron))
            list_out_pred = make_prediction_sig(predictions,list_output_weights[j])
        if np.mean(list_out_pred) >= 0.5:
            pred = 1
        else:
            pred = 0
        error = class_test[i] - pred
        sum_error += error**2
    print("Bagging nbrPMC=%d nbrNeurons=%d epoch=%d error=%d false=%.3f%%" %(nbr_of_pmc,nbr_neurons,epoch,sum_error, sum_error * 100 / len(data_test)))
    return {'nbrPMC':nbr_of_pmc,'nbr_neurons':nbr_neurons, 'epoch':epoch,'error':sum_error,'false':sum_error * 100 / len(data_test)}
                
def test_training_sig(data,classes, hiden_neurons,output_weights, epoch, rate):
    sum_error = 0.0
    for i,row in enumerate(data):
            predictions = []
            for neuron in hiden_neurons:
                predictions.append(make_prediction_sig(row, neuron))
            out_pred = make_prediction_sig_bool(predictions,output_weights)
            error = classes[i] - out_pred
            sum_error += error**2
    print("Test bands nbrneurons=%d epoch=%d rate=%.3f error=%d false=%.3f%%" %(len(hiden_neurons),epoch,rate,sum_error, sum_error * 100 / len(data)))
    return {'nbr_neurons':len(hiden_neurons), 'epoch':epoch,'rate':rate,'error':sum_error,'false':sum_error * 100 / len(data_test)}

def big_test(data_train,classes_train,datas_test,classes_test):
    epoch_list = [10,50,100,200,500,1000]
    nbr_neurons_list = [10,50,100,200,500,1000]
    list_rate = [0.1]
    results_PMC = []
    for rate in list_rate:
        for nbr_neurons in nbr_neurons_list:
            for epoch in epoch_list:
                hiden_neurons,output_weights = train_weights_multi(data_train, classes_train, nbr_neurons, rate,epoch)
                results_PMC.append(test_training_sig(datas_test,classes_test,hiden_neurons, output_weights,epoch, rate))
    return results_PMC

    
if __name__ == "__main__":
    data_train,class_train = open_file_bands("ressource/DATA/bands.dat")
    bagging_results = []
    PMC_results = []
    for i in range(3):
        data_t, data_test, class_t, class_test = train_test_split(data_train, class_train, test_size=0.1, random_state=0)
        PMC_results.append(big_test(data_t,class_t,data_test,class_test))
    with open('outPMC.txt', 'a') as f:
        for data in PMC_results:
            for row in data:
                f.write(json.dumps(row)+'\n')
