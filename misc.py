import torch

def find_all_substr(string, substr):
    m = len(string)
    n = len(substr)
    indices = list()
    
    for i in range(m - n + 1):
        if string[i:(i + n)] == substr:
            indices.append(i)
    return indices

def check_prefix(string, prefix):
    if len(string) < len(prefix):
        return False
    return string[:len(prefix)] == prefix

def get_collection_dim(collection, depth=0):
    t = type(collection)
    if t in [list, tuple]:
        print(f"{depth * "--"}{len(collection)}")
        for obj in collection:
            get_collection_dim(obj, depth=depth + 1)
    elif t == torch.Tensor:
        print(f"{depth * "--"}{collection.shape}")