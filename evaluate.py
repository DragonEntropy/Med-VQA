import regex as re
import argparse

def check_prefix(string, prefix):
    if len(string) < len(prefix):
        return False
    return string[:len(prefix)] == prefix

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
    print(string)
    return False


def get_score_old(answer, keywords):
    hits = 0
    for keyword in keywords:
        hits += (keyword in answer)
    if hits == 0:
        print(keywords, answer)
    return hits, len(keywords)

def get_score(answer, keywords):
    cleaned_answer, _ = re.subn("""[.,?!'"]""", '', answer)
    tokens = cleaned_answer.split()
    hits = 0
    for keyword in keywords:
        hits += (keyword in tokens)
    #if tokens[-1] == tokens[-2] and tokens[-2] == tokens[3]:
    #    return 0, 0
    return hits, len(keywords)


def evaluate(pred_file, true_file, output_file=None, k=0, direct=False):
    dialogue_step = (0, 0)
    correct_list = list()
    total_list = list()
    keywords = None
    current_answer = None
    hallucination_count = 0

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
        print(current_line)
        if check_prefix(current_line, "<entry start>"):
            if dialogue_step == (4, k):
                hits, total = get_score(current_answer, keywords)
                correct_list.append(hits)
                total_list.append(total)
                if check_hallucination(current_answer):
                    hallucination_count += 1

            true = true_file.__next__()
            keywords = true.lower().replace("\n", "").split(", ")
            current_answer = ""
            dialogue_step = (1, 0)

        elif check_prefix(current_line, "user") or check_prefix(current_line, " user"):
            if dialogue_step[0] == 1:
                dialogue_step = (2, dialogue_step[1])
            elif dialogue_step[0] == 4:
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
        print(dialogue_step)

    averages = [hits / total for (hits, total) in zip(correct_list, total_list)]
    micro_average = sum(correct_list) / sum(total_list)
    macro_average = sum(averages) / len(total_list)
    print(f"Micro average: {(100 * micro_average):.2f}%")
    print(f"Macro average: {(100 * macro_average):.2f}%")
    print(f"Hallucination rate: {(100 * hallucination_count / len(total_list)):.2f}%")

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-p", "--pred", type=str, default="alex/results/pred.txt")
    argparser.add_argument("-t", "--true", type=str, default="alex/results/ans.txt")
    argparser.add_argument("-o", "--output", type=str, default="alex/results/ans_summary.txt")
    argparser.add_argument("-k", "--kshot", type=int, default=0)
    argparser.add_argument("-d", "--direct", action="store_true")
    
    args = argparser.parse_args()
    pred_path = args.pred
    true_path = args.true
    output_path = args.output
    k = args.kshot
    direct = args.direct

    with open(pred_path, 'r') as pred_file, open(true_path, 'r') as true_file, open(output_path, 'w') as output_file:
        evaluate(pred_file, true_file, output_file=output_file, k=k, direct=direct)

if __name__ == "__main__":
    main()