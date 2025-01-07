import os
import argparse

from llava_model import LlavaModel, LlavaInterleaveModel
from qwen_model import QwenModel
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
    argparser.add_argument("-e", "--exampleimage", action="store_false")
    argparser.add_argument("-k", "--kshot", type=int, default=1)
    argparser.add_argument("-b", "--batchsize", type=int, default=1)
    argparser.add_argument("-m", "--model", type=str, default="L")
    argparser.add_argument("-c", "--count", type=int, default=-1)
    args = argparser.parse_args()

    model = None
    if args.model == "L":
        print("Running with Llava model")
        model = LlavaModel(args.srcpath, args.imagepath)
    elif args.model == "LI":
        print("Running with Llava interleave model")
        model = LlavaInterleaveModel(args.srcpath, args.imagepath)
    elif args.model == "Q":
        print("Running with Qwen model")
        model = QwenModel(args.srcpath, args.imagepath)

    else:
        print("An invalid model was specified")
        return
    
    with open(args.outputpath, 'w') as output_file, open(args.answerpath, 'w') as answer_file:
        pipeline = None
        if args.direct:
            print("Running with direct prompting")
            pipeline = DirectPipeline(model, output_file, answer_file, count=args.count, batch_size=args.batchsize)
        elif args.zeroshot:
            print("Running with zero-shot prompting")
            pipeline = ZeroShotPipeline(model, output_file, answer_file, count=args.count, batch_size=args.batchsize)
        elif args.fewshot:
            print("Running with few-shot prompting")
            pipeline = FewShotPipeline(model, output_file, answer_file, count=args.count, batch_size=args.batchsize, example_image=args.exampleimage, k=args.kshot)
        else:
            print("You must specify a type of prompting to use (--direct, --zeroshot, --fewshot, etc)")
            return
        
        pipeline.run()

if __name__ == "__main__":
    main()