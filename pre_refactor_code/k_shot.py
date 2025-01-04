from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlamaForCausalLM
import os
import json

def generate_initial_prompts(
    data,
    image_path,
    processor,
    batch_size=1,
    CoT_prompt="Let's think step by step.",
    example_question="",
    example_response="",
    example_image_path=""
):
    prompts = list()
    images = list()
    true_answers = list()

    current_size = 0
    for entry in data:
        if entry["q_lang"] != "en":
            continue

        current_size += 1
        question = entry["question"]
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text":  f"{example_question}\n"},
                    {"type": "image"}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"{example_response}\n"},
                ]
            },
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

        prompt = processor.apply_chat_template(conversation, history=[], add_generation_prompt=False)
        image_url = os.path.join(image_path, entry["img_name"])
        image = Image.open(image_url)
        example_image_url = os.path.join(image_path, example_image_path)
        example_image = Image.open(example_image_url)

        prompts.append(prompt)
        images.append([image, example_image])
        true_answers.append(entry["answer"])

        if current_size == batch_size:
            yield prompts, images, true_answers
            current_size = 0
            prompts = list()
            images = list()
            true_answers = list()

    if len(prompts) > 0:
        yield prompts, images, true_answers

def generate_final_prompts(
    outputs,
    processor,
    answer_prompt="The answer is",
    example_image_path=""
):
    final_prompts = list()
    for output in outputs:
        output = output[:6] + "<image>" + output[7:]
        index = output.find("\nUSER:")
        output = output[:(index + 6)] + "<image>" + output[(index + 7):]
        final_prompts.append(f"{output}\n{answer_prompt}")

    return final_prompts

def run_model(model, processor, prompts, images, batch_size=1):
    outputs = list()
    n_prompts = min(len(prompts), len(images))
    n_batches = 1 + (n_prompts - 1) // batch_size

    for i in range(n_batches):
        lower = i * batch_size
        upper = min(lower + batch_size, n_prompts)
        batch = processor(
            images=images[lower:upper][0],
            text=prompts[lower:upper][0],
            return_tensors='pt',
            padding=True
        ).to(0, torch.float16)

        raw_outputs = model.generate(**batch, max_new_tokens=200)
        for raw_output in raw_outputs:
            outputs.append(processor.decode(raw_output, skip_special_tokens=True))

    return outputs

def pipeline(model, processor, output_file, answer_file, data, image_path):
    example_question="What modality is used to take this image?"
    example_response="""Let's think step by step.
1. The image is a medical image, which suggests that it is related to healthcare or anatomy.
2. The image is a cross-sectional view of the human body, which indicates that it is likely an X-ray or CT scan.
3. The image is black and white, which is a common characteristic of medical images.

Based on these observations, the modality used to take this image is most likely an X-ray or a CT scan.
The answer is a CT scan\n"""
    example_image_path="xmlab102/source.jpg"

    initial_prompts = generate_initial_prompts(
        data,
        image_path,
        processor,
        example_question=example_question,
        example_response=example_response,
        example_image_path=example_image_path
    )
    for i, (prompts, images, true_answers) in enumerate(initial_prompts):
        print(f"Running batch {i + 1}")

        # Generating CoT responses from initial prompts
        print('Running model for CoT step')
        outputs = run_model(model, processor, prompts, images)

        # Generating final prompts
        print("Generating final prompts from CoT outputs")
        final_prompts = generate_final_prompts(outputs, processor, example_image_path=example_image_path)

        # Getting final answers
        print("Running model for final answers")
        outputs = run_model(model, processor, final_prompts, images)

        for output in outputs:
            output_file.write(f"<ENTRY START>\n{output}\n\n")
        for answer in true_answers:
            answer_file.write(f"{answer}\n")


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
    # model_id = "PanaceaAI/BiomedGPT-Base-Pretrained"
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

    with open("alex/results/test_few.txt", 'w') as output_file, open("alex/results/ans_few.txt", 'w') as answer_file, open(data_path, 'r') as data_file:
        data = json.load(data_file)
        pipeline(model, processor, output_file, answer_file, data, image_path)


if __name__ == "__main__":
    main()
