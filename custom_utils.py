import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder



def load_data(root_dir, num_files):
    # load the data into a dataframe
    X = []
    Y = []

    for i in range(num_files):
        text_file_path = root_dir + "/" + str(i) + "_word.txt"
        text_file = open(text_file_path, "r")
        emg_file_path = root_dir + "/" + str(i) + "_emg.npy"
        emg_file = np.load(emg_file_path)
        X.append(emg_file)
        Y.append(text_file.read())

    # do one hot encoding on Y 
    Y = np.array(Y)
    Y = Y.reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown="ignore")
    enc.fit(Y)
    Y = enc.transform(Y).toarray()

    max_dimension = max(arr.shape[0] for arr in X)
    # pad the arrays to be the same size
    X = np.array([np.pad(arr, ((0, max_dimension - arr.shape[0]), (0, 0)), 'constant') for arr in X])
    
    return X,Y


load_data("single_word_mouthed", 150)