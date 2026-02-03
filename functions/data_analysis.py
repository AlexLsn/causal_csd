import numpy as np

import pickle
import os
import glob



def load_and_rename_files(directory, pattern):
    search_pattern = os.path.join(directory, pattern)
    files = glob.glob(search_pattern)
    data_dict = {}
    
    for file_path in files:
        file_name = os.path.basename(file_path).replace('.dat', '')
        new_name = file_name.replace('-', '_').replace('0.001', '0001').replace('0.005', '0005').replace('0.01', '001')
        
        with open(file_path, 'rb') as file:
            data_dict[new_name] = pickle.load(file)
    
    return data_dict



def truncate_dict_values(d, length=200):
    """
    Truncate values in the nested dictionaries only if their length exceeds the given threshold.

    Args:
        d (dict): The dictionary containing nested dictionaries.
        length (int): Maximum allowed length of each key's value.

    Returns:
        dict: A new dictionary with conditionally truncated values.
    """
    truncated = {}
    for key, sub_dict in d.items():
        truncated[key] = {
            k: v[:length] if len(v) > length else v
            for k, v in sub_dict.items()
        }
    return truncated