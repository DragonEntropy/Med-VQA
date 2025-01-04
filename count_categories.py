import os
import json
from collections import defaultdict

def accumulate_categories(data):
    categories = defaultdict(lambda: 0)
    for entry in data:
        if entry["q_lang"] == "en":
            categories[entry["answer"].lower()] += 1
    categories = sorted(categories)
    print(categories)
    print(categories.keys())
    return categories

def main():
    cwd = os.path.dirname(os.getcwd())
    data_path = os.path.join(cwd, "Downloads", "Slake", "test.json")
    with open(data_path, 'r') as data_file:
        data = json.load(data_file)
        accumulate_categories(data)

if __name__ == "__main__":
    main()