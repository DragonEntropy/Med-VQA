import matplotlib.pyplot as plt
import regex as re

def check_prefix(string, prefix):
    if len(string) < len(prefix):
        return False
    return string[:len(prefix)] == prefix

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

def evaluate(pred_file, true_file, output_file=None):
    dialogue_step = 0
    correct_list = list()
    total_list = list()
    keywords = None
    current_answer = None

    for current_line in pred_file:
        """
        Dialogue steps:
            0: Initially searching for "<ENTRY START>"
            1: Searching for "USER:" initial prompt
            2: Searching for " ASSISTANT:" CoT response
            3: Searching for "USER:" final prompt
            4: Searching for " ASSISTANT:" final response
            5: Scanning final response
        """
        if check_prefix(current_line, "<ENTRY START>"):
            if dialogue_step == 3:
                hits, total = get_score(current_answer.lower(), keywords)
                correct_list.append(hits)
                total_list.append(total)

                if output_file:
                    output_file.write(f"Predicted response: {current_answer}\n")
                    output_file.write(f"True answers: {keywords}\n")
                    output_file.write(f"Score: {hits}/{total}\n\n")

            true = true_file.__next__()
            keywords = true.lower().replace("\n", "").split(", ")
            current_answer = ""
            dialogue_step = 1

        elif dialogue_step == 3:
            current_answer = "".join((current_answer, current_line))

        elif check_prefix(current_line, "USER:"):
            if dialogue_step == 1:
                dialogue_step = 2

        elif check_prefix(current_line, " ASSISTANT:"):
            if dialogue_step == 2:
                dialogue_step = 3

    averages = [hits / total for (hits, total) in zip(correct_list, total_list)]
    micro_average = sum(correct_list) / sum(total_list)
    macro_average = sum(averages) / len(total_list)
    print(f"Micro average: {(100 * micro_average):.2f}%")
    print(f"Macro average: {(100 * macro_average):.2f}%")
    plt.hist(averages, 20, (0, 1))
    plt.show()


def evaluate_split(pred_file, true_file):
    dialogue_step = 0
    correct_lists = [list(), list()]
    total_lists = [list(), list()]
    closed_set = ["yes", "no"]
    keywords = None
    current_answer = None

    for current_line in pred_file:
        """
        Dialogue steps:
            0: Initially searching for "<ENTRY START>"
            1: Searching for "USER:" initial prompt
            2: Searching for " ASSISTANT:" CoT response
            3: Searching for "USER:" final prompt
            4: Searching for " ASSISTANT:" final response
            5: Scanning final response
        """
        if check_prefix(current_line, "<ENTRY START>"):
            if dialogue_step == 3:
                hits, total = get_score(current_answer.lower(), keywords)
                correct_lists[keywords[0] in closed_set].append(hits)
                total_lists[keywords[0] in closed_set].append(total)

            true = true_file.__next__()
            keywords = true.lower().replace("\n", "").split(", ")
            current_answer = ""
            dialogue_step = 1

        elif dialogue_step == 3:
            current_answer = "".join((current_answer, current_line))

        elif check_prefix(current_line, "USER:"):
            if dialogue_step == 1:
                dialogue_step = 2

        elif check_prefix(current_line, " ASSISTANT:"):
            if dialogue_step == 2:
                dialogue_step = 3

    open_correct = correct_lists[0]
    closed_correct = correct_lists[1]
    open_totals = total_lists[0]
    closed_totals = total_lists[1]
    averages = [hits / total for (hits, total) in zip(open_correct, open_totals)]
    micro_average = sum(open_correct) / sum(open_totals)
    macro_average = sum(averages) / len(open_totals)
    print(f"Micro average: {(100 * micro_average):.2f}%")
    print(f"Macro average: {(100 * macro_average):.2f}%")
    plt.hist(averages, 20, (0, 1))
    plt.show()

    averages = [hits / total for (hits, total) in zip(closed_correct, closed_totals)]
    micro_average = sum(closed_correct) / sum(closed_totals)
    macro_average = sum(averages) / len(closed_totals)
    print(f"Micro average: {(100 * micro_average):.2f}%")
    print(f"Macro average: {(100 * macro_average):.2f}%")

def main():
    pred_path = "alex/results/test_few.txt"
    true_path = "alex/results/ans_few.txt"
    output_path = "alex/results/ans_summary.txt"

    with open(pred_path, 'r') as pred_file, open(true_path, 'r') as true_file, open(output_path, 'w') as output_file:
        evaluate(pred_file, true_file, output_file=output_file)
        pred_file.readlines()

if __name__ == "__main__":
    main()