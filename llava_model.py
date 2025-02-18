from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import os

from model import Model
from misc import find_all_substr


class LlavaModel(Model):
    model_id = "llava-hf/llava-1.5-7b-hf"
    CoT_prompt = "Let's think step by step."
    answer_prompt = "So, the answer"

    def __init__(self, data_path, image_path):
        super().__init__(data_path, image_path)

        self.base_model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir
        ).to(0)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def provide_initial_prompts(self, data, examples=[], batch_size=1, direct=False, example_image=True, max=-1):
        count = 0
        prompts = list()
        images = list()
        true_answers = list()

        # Examples is a list of (question, image, response) tuples
        for entry in data:
            if count == max:
                break

            # Check language
            if entry["q_lang"] != "en":
                continue

            # Adding examples to conversation
            conversation = list()

            examples_source = examples
            if type(examples) == dict:
                category = entry["content_type"]
                examples_source = examples[category]

            for example in examples_source:
                conversation.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text":  f"Here is an example of the answer format I want you to follow.\n{example[0]}\n"}
                    ]
                })
                if example_image:
                    conversation[-1]["content"].append({"type": "image"})
                conversation.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"{example[2]}\n"},
                    ]
                })

            # Adding required question to conversation
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text":  f"Please answer the following question.\n{entry["question"]}\n"},
                    {"type": "image"}
                ]
            }),

            # Skip the chain of thought step if directly questioning
            if direct:
                conversation.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ""},
                    ]
                })
            else:
                conversation.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"{self.CoT_prompt}"},
                    ]
                })

            # Generate prompt via chat template
            prompt = self.processor.apply_chat_template(conversation, history=[], add_generation_prompt=False)
            prompts.append(prompt)

            # Combine example images with the question's final image
            if example_image:
                for example in examples_source:
                    image_url = os.path.join(self.image_path, example[1])
                    image = Image.open(image_url)
                    images.append(image)

            image_url = os.path.join(self.image_path, entry["img_name"])
            image = Image.open(image_url)
            images.append(image)

            true_answers.append(entry["answer"])
            count += 1

            if count % batch_size == 0:
                yield prompts, images, true_answers
                prompts = list()
                images = list()
                true_answers = list()
        
        # Yield final batch when data is empty
        if count % batch_size > 0:
            yield prompts, images, true_answers

    def generate_final_prompts(self, outputs, example_image=True):
        # Instead of generating a new conversation template, alter existing outputs
        final_prompts = list()
        for output in outputs:
            output = self.image_injection_method(output)
            final_prompts.append(output)
        return final_prompts
    
    def run_model(self, prompts, images, store_attention=False):
        raw_input = self.processor(
            images=images,
            text=prompts,
            return_tensors='pt',
            padding=True
        ).to(0, torch.float16)

        outputs = list()
        raw_outputs = self.base_model.generate(**raw_input, max_new_tokens=200)
        for raw_output in raw_outputs:
            outputs.append(self.processor.decode(raw_output, skip_special_tokens=True))
        return outputs
    
    def image_injection_method(self, prompt, example_image=True):
        indices = find_all_substr(prompt, "USER:")
        indices.reverse()
        for j, i in enumerate(indices):
            if example_image or j == 0:
                prompt = prompt[:(i + 6)] + "<image>" + prompt[(i + 6):]
        prompt += f"\n{self.answer_prompt}"
        
        return prompt
    
class LlavaInterleaveModel(LlavaModel):
    def __init__(self, data_path, image_path):
        self.model_id = "llava-hf/llava-interleave-qwen-7b-hf"
        self.answer_prompt = "So, "
        super().__init__(data_path, image_path)

    def image_injection_method(self, prompt, example_image=True):
        indices = find_all_substr(prompt, "user")
        indices.reverse()
        for j, i in enumerate(indices):
            if example_image or j == 0:
                if j == len(indices) - 1:
                    prompt = prompt[:i] + "<|im_start|>" + prompt[:(i + 5)] + "<image>" + prompt[(i + 5):]
                else:
                    prompt = prompt[:i] + "<|im_end|>\n<|im_start|>" + prompt[:(i + 5)] + "<image>\n" + prompt[(i + 5):]
            else:
                if j == len(indices) - 1:
                    prompt = prompt[:i] + "<|im_start|>" + prompt[i:]
                else:
                    prompt = prompt[:i] + "<|im_end|>\n<|im_start|>" + prompt[i:]

        indices = find_all_substr(prompt, "assistant")
        indices.reverse()
        for i in indices:
            prompt = prompt[:i] + "<|im_end|>\n<|im_start|>" + prompt[i:]
        prompt += f"\n{self.answer_prompt}<|im_end|>"

        return prompt
