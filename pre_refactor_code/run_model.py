import argparse
import os

import direct_prompting
import zero_shot
import k_shot

def main():
    cwd = os.path.dirname(os.getcwd())
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-s", "--srcpath", type=str, default=os.path.join(cwd, "Downloads", "Slake", "test.json"))
    argparser.add_argument("-i", "--imagepath", type=str, default=os.path.join(cwd, "Downloads", "Slake", "imgs"))
    argparser.add_argument("-o", "--outputpath", type=str, default=os.path.join(cwd, "Documents", "alex", "outputs"))
    argparser.add_argument("-z", "--zeroshot", action="store_true")
    argparser.add_argument("-f", "--fewshot", action="store_true")
    argparser.add_argument("-d", "--direct", action="store_true")
    args = argparser.parse_args()

    scripts = [
        direct_prompting,
        zero_shot,
        k_shot
    ]

    script_index = -1
    if args.direct:
        script_index = 0
    elif args.zeroshot:
        script_index = 1
    elif args.fewshot:
        script_index = 2
    if script_index > 0:
        scripts[script_index].main()
    else:
        print("You must specify a type of prompting to use (--direct, --zeroshot, --fewshot, etc)")

if __name__ == "__main__":
    main()