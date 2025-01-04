import os
import argparse

from model import LlavaModel
from pipeline import DirectPipeline, ZeroShotPipeline, FewShotPipeline

def main():
    cwd = os.path.dirname(os.getcwd())
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-s", "--srcpath", type=str, default=os.path.join(cwd, "Downloads", "Slake", "test.json"))
    argparser.add_argument("-i", "--imagepath", type=str, default=os.path.join(cwd, "Downloads", "Slake", "imgs"))
    argparser.add_argument("-o", "--outputpath", type=str, default=os.path.join(cwd, "Documents", "alex", "results", "pred.txt"))
    argparser.add_argument("-a", "--answerpath", type=str, default=os.path.join(cwd, "Documents", "alex", "results", "ans.txt"))
    argparser.add_argument("-z", "--zeroshot", action="store_true")
    argparser.add_argument("-f", "--fewshot", action="store_true")
    argparser.add_argument("-d", "--direct", action="store_true")
    args = argparser.parse_args()

    model = LlavaModel(args.srcpath, args.imagepath)
    with open(args.outputpath, 'w') as output_file, open(args.answerpath, 'w') as answer_file:
        pipeline = None
        if args.direct:
            print("Running with direct prompting")
            pipeline = DirectPipeline(model, output_file, answer_file)
        elif args.zeroshot:
            print("Running with zero-shot prompting")
            pipeline = ZeroShotPipeline(model, output_file, answer_file)
        elif args.fewshot:
            print("Running with few-shot prompting")
            pipeline = FewShotPipeline(model, output_file, answer_file, k=1)
        else:
            print("You must specify a type of prompting to use (--direct, --zeroshot, --fewshot, etc)")
            return
        
        pipeline.run()

if __name__ == "__main__":
    main()