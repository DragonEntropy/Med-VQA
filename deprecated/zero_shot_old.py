from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os
import json

def generate_initial_prompts(
    data,
    image_path,
    processor,
    CoT_prompt="Let's think step by step."
):
    prompts = list()
    images = list()
    true_answers = list()

    for i, entry in enumerate(data):
        question = entry["question"]
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text":  f"{question}\n"},
                    {"type": "image"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"{CoT_prompt}"},
                ]
            }
        ]

        prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)
        image_url = os.path.join(image_path, entry["img_name"])
        image = Image.open(image_url)

        prompts.append(prompt)
        images.append(image)
        true_answers.append(entry["answer"])
        
    return prompts, images, true_answers

def generate_final_prompts(
    outputs,
    processor,
    answer_prompt="What is your final answer?"
):
    final_prompts = list()
    for output in outputs:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text":  f"{answer_prompt}\n"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": ""}
                ]
            }
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=False)

        # Manually insert <image> tag into previous output
        output = output[:6] + "<image>" + output[7:]
        final_prompts.append(f"{output}\n{prompt}")

    return final_prompts

def run_model(model, processor, prompts, images, batch_size=10):
    outputs = list()
    n_prompts = min(len(prompts), len(images))
    n_batches = 1 + (n_prompts - 1) // batch_size

    for i in range(n_batches):
        print(f"Running batch {i + 1} of {n_batches}")
        lower = i * batch_size
        upper = min(lower + batch_size, n_prompts)
        batch = processor(
            images=images[lower:upper],
            text=prompts[lower:upper],
            return_tensors='pt',
            padding=True
        ).to(0, torch.float16)

        raw_outputs = model.generate(**batch, max_new_tokens=200)
        for raw_output in raw_outputs:
            outputs.append(processor.decode(raw_output, skip_special_tokens=True))

    return outputs

def main():
    # Data
    cwd = os.path.dirname(os.getcwd())
    data_path = os.path.join(cwd, "Downloads", "Slake", "test.json")
    image_path = os.path.join(cwd, "Downloads", "Slake", "imgs")
    model_path = os.path.join(cwd, "Downloads", "llava-v1.5-13b-hf")
    #data_path = os.path.join(cwd, "tutorials", "vqa", "data", "slake", "anno-close", "test.json")
    #image_path = os.path.join(cwd, "tutorials", "vqa", "data", "slake", "images")

    # Model
    print("Loading model")
    model_id = "llava-hf/llava-1.5-7b-hf"
    """
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    """
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
        #model_kwargs={"quantization_config": config}
    ).to(0)
    processor = AutoProcessor.from_pretrained(model_id)
    print("Loading complete!")

    # Generating initial prompts
    print("Preprocessing data")
    prompts, images, true_answers = None, None, None
    with open(data_path, 'r') as data_file:
        data = json.load(data_file)
        prompts, images, true_answers = generate_initial_prompts(data, image_path, processor)
    if not (prompts and images and true_answers):
        print("Error loading input data")
        return
    print(prompts[0])

    # Generating CoT responses from initial prompts
    print('Running model for CoT step')
    outputs = run_model(model, processor, prompts, images)

    # Generating final prompts
    print("Generating final prompts from CoT outputs")
    final_prompts = generate_final_prompts(outputs, processor)

    # Getting final answers
    print("Running model for final answers")
    outputs = run_model(model, processor, final_prompts, images)

    with open("alex/results.txt", 'w') as output_file:
        for output in outputs:
            output_file.write(f"<ENTRY START>\n{output}\n\n")


if __name__ == "__main__":
    main()