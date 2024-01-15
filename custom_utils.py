from pandas import pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

def load_data(root_dir, num_files):
    df = pd.DataFrame()
    for i in range(num_files):
        text_file_path = root_dir + "/" + str(i) + "_word.txt"
        text_file = open(text_file_path, "r")
        emg_file_path = root_dir + "/" + str(i) + "_emg.npy"
        emg_file = np.load(emg_file_path)

        df = df.append(pd.DataFrame({"emg": emg_file, "word": text_file.read()}))
    
    # create a new column with the one-hot encoded labels for text
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(df[["word"]])
    one_hot = enc.transform(df[["word"]]).toarray()
    df["one_hot"] = list(one_hot)
    
    return df