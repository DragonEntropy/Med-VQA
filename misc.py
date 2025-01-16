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