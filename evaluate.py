import regex as re
import argparse
from misc import check_prefix
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet', download_dir="/mnt/HDD3/vri/Applications/anaconda3/lib/nltk_data")
import os
import json

def check_hallucination(string, m=10, n=4):
    # Check for 1 character responses
    prefix = "so, "
    if check_prefix(string, prefix):
        if len(string) - len(prefix) <= 3:
            return True

    # Check for n repeated substrings up to length m
    current_count = 0
    for sublen in range(1, m + 1):
        current_substr = None
        for i in range(len(string) // sublen):
            substr = string[sublen * i : sublen * (i + 1)]
            if current_substr == substr:
                current_count += 1
                if current_count >= n:
                    return True
            else:
                current_substr = substr
                current_count = 1
    return False


def get_score_old(answer, keywords):
    hits = 0
    for keyword in keywords:
        hits += (keyword in answer)
    if hits == 0:
        print(keywords, answer)
    return hits, len(keywords)

def get_score(answer, keywords, lem):
    cleaned_answer, _ = re.subn("""[.,?!'"*]""", '', answer)
    tokens = [lem.lemmatize(token) for token in cleaned_answer.split()]
    hits = 0
    for keyword in keywords:
        hits += (lem.lemmatize(keyword) in tokens)
    #if tokens[-1] == tokens[-2] and tokens[-2] == tokens[3]:
    #    return 0, 0
    return hits, len(keywords)


def evaluate(pred_file, true_file, k=0, direct=False):
    lem = WordNetLemmatizer()

    dialogue_step = (0, 0)
    correct_list = list()
    total_list = list()
    keywords = None
    current_answer = None
    hallucination_count = 0
    question_number = -1
    blacklist = [31, 125]

    for current_line in pred_file:
        current_line = current_line.lower()
        """
        Dialogue steps:
            0: Initially searching for "<ENTRY START>"
            1: Searching for "USER:" prompt
            2: Searching for " ASSISTANT:" CoT response
            3: Searching for final response
            4: Scanning final response
        """
        if check_prefix(current_line, "<entry start>"):
            if dialogue_step == (4, k):
                hits, total = get_score(current_answer, keywords, lem)
                correct_list.append(hits)
                total_list.append(total)
                if check_hallucination(current_answer):
                    hallucination_count += 1
                if question_number % 100 == 99:
                    print(sum(correct_list) / sum(total_list))

            while dialogue_step != (1, 0) or question_number in blacklist:
                true = true_file.__next__()
                keywords = true.lower().replace("\n", "").split(", ")
                current_answer = ""
                dialogue_step = (1, 0)
                question_number += 1

        elif check_prefix(current_line, "user") or check_prefix(current_line, " user"):
            if dialogue_step[0] == 1:
                dialogue_step = (2, dialogue_step[1])
            elif dialogue_step[0] == 4:
                dialogue_step = (2, dialogue_step[1] + 1)
            elif dialogue_step[0] == 3:
                dialogue_step = (2, dialogue_step[1] + 1)

        elif check_prefix(current_line, "assistant") or check_prefix(current_line, " assistant"):
            if dialogue_step[0] == 2:
                if direct:
                    dialogue_step = (4, dialogue_step[1])
                else:
                    dialogue_step = (3, dialogue_step[1])

        elif check_prefix(current_line, "so, "):
            if dialogue_step[0] == 3:
                dialogue_step = (4, dialogue_step[1])
                
        if dialogue_step == (4, k):
            current_answer = "".join((current_answer, current_line))

    averages = [hits / total for (hits, total) in zip(correct_list, total_list)]
    micro_average = sum(correct_list) / sum(total_list)
    macro_average = sum(averages) / len(total_list)
    print(f"Total score: {sum(correct_list)} / {sum(total_list)}")
    print(f"Macro average: {(100 * macro_average):.2f}%")
    print(f"Micro average: {(100 * micro_average):.2f}%")
    print(f"Hallucination rate: {(100 * hallucination_count / len(total_list)):.2f}%")


def get_next_category(true_data, category):
    entry = true_data.__next__()
    while entry["q_lang"] != "en" or entry["content_type"] != category:
        entry = true_data.__next__()
    return entry["answer"]


def evaluate_by_category(data_file, pred_path, k=0, direct=False):
    lem = WordNetLemmatizer()

    categories = ["Abnormality", "Color", "KG", "Modality", "Organ", "Plane", "Position", "Quantity", "Shape", "Size"]
    data_json = json.load(data_file)

    for category in categories:
        
        dialogue_step = (0, 0)
        correct_list = list()
        total_list = list()
        keywords = None
        current_answer = None
        hallucination_count = 0
        question_number = -1
        blacklist = [] # [31, 125]
        true_data = iter(data_json)

        with open(os.path.join(pred_path, f"{category.lower()}.txt"), 'r') as pred_file:
            for current_line in pred_file:
                current_line = current_line.lower()
                """
                Dialogue steps:
                    0: Initially searching for "<ENTRY START>"
                    1: Searching for "USER:" prompt
                    2: Searching for " ASSISTANT:" CoT response
                    3: Searching for final response
                    4: Scanning final response
                """
                if check_prefix(current_line, "<entry start>"):
                    if dialogue_step == (4, k):
                        hits, total = get_score(current_answer, keywords, lem)
                        correct_list.append(hits)
                        total_list.append(total)
                        if check_hallucination(current_answer):
                            hallucination_count += 1

                    while dialogue_step != (1, 0) or question_number in blacklist:
                        true = get_next_category(true_data, category)
                        keywords = true.lower().replace("\n", "").split(", ")
                        current_answer = ""
                        dialogue_step = (1, 0)
                        question_number += 1

                elif check_prefix(current_line, "user") or check_prefix(current_line, " user"):
                    if dialogue_step[0] == 1:
                        dialogue_step = (2, dialogue_step[1])
                    elif dialogue_step[0] == 4:
                        dialogue_step = (2, dialogue_step[1] + 1)
                    elif dialogue_step[0] == 3:
                        dialogue_step = (2, dialogue_step[1] + 1)

                elif check_prefix(current_line, "assistant") or check_prefix(current_line, " assistant"):
                    if dialogue_step[0] == 2:
                        if direct:
                            dialogue_step = (4, dialogue_step[1])
                        else:
                            dialogue_step = (3, dialogue_step[1])

                elif check_prefix(current_line, "so, "):
                    if dialogue_step[0] == 3:
                        dialogue_step = (4, dialogue_step[1])
                        
                if dialogue_step == (4, k):
                    current_answer = "".join((current_answer, current_line))

            averages = [hits / total for (hits, total) in zip(correct_list, total_list)]
            micro_average = sum(correct_list) / sum(total_list)
            macro_average = sum(averages) / len(total_list)
            print(f"Printing results for category {category}")
            print(f"Total score: {sum(correct_list)} / {sum(total_list)}")
            print(f"Macro average: {(100 * macro_average):.2f}%")
            print(f"Micro average: {(100 * micro_average):.2f}%")
            print(f"Hallucination rate: {(100 * hallucination_count / len(total_list)):.2f}%")


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-p", "--pred", type=str, default="alex/results/pred.txt")
    argparser.add_argument("-t", "--true", type=str, default="alex/results/ans_const.txt")
    argparser.add_argument("-k", "--kshot", type=int, default=0)
    argparser.add_argument("-d", "--direct", action="store_true")
    
    args = argparser.parse_args()
    pred_path = args.pred
    true_path = args.true
    k = args.kshot
    direct = args.direct

    with open(pred_path, 'r') as pred_file, open(true_path, 'r') as true_file:
        evaluate(pred_file, true_file, k=k, direct=direct)


def category_main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-p", "--pred", type=str, default="alex/results/auto_categorical")
    argparser.add_argument("-t", "--true", type=str, default="../Downloads/Slake/test.json")
    argparser.add_argument("-k", "--kshot", type=int, default=0)
    argparser.add_argument("-d", "--direct", action="store_true")
    
    args = argparser.parse_args()
    pred_path = args.pred
    true_path = args.true
    k = args.kshot
    direct = args.direct

    with open(true_path, 'r') as true_file:
        evaluate_by_category(true_file, pred_path, k=k, direct=direct)

if __name__ == "__main__":
    category_main()