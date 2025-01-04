from transformers import LlavaForConditionalGeneration, AutoProcessor
import torch
import os
from PIL import Image
from abc import ABC, abstractmethod

from misc import find_all_substr

class Model(ABC):
    def __init__(self, data_path, image_path):
        self.data_path = data_path
        self.image_path = image_path
        self.base_model = None
        self.processor = None

        self.examples = [(
            "What modality is used to take this image?",
            "xmlab102/source.jpg",
            """Let's think step by step.
1. The image is a medical image, which suggests that it is related to healthcare or anatomy.
2. The image is a cross-sectional view of the human body, which indicates that it is likely an X-ray or CT scan.
3. The image is black and white, which is a common characteristic of medical images.

Based on these observations, the modality used to take this image is most likely an X-ray or a CT scan.
The answer is a CT scan\n"""
        )]

    @abstractmethod
    def provide_initial_prompts(self, data, examples=[], direct=False, max=-1):
        yield None
    
    @abstractmethod
    def generate_final_prompt(self, output):
        return None
    
    @abstractmethod
    def run_model(self, prompt, images):
        return None


class LlavaModel(Model):
    model_id = "llava-hf/llava-1.5-7b-hf"
    CoT_prompt = "Let's think step by step."
    answer_prompt = "So,"

    def __init__(self, data_path, image_path):
        super().__init__(data_path, image_path)

        self.base_model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(0)
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def provide_initial_prompts(self, data, examples=[], direct=False, max=-1):
        # Combine example images with blank final image
        images = list()
        for example in examples:
            image_url = os.path.join(self.image_path, example[1])
            image = Image.open(image_url)
            images.append(image)
        # Last space is reserved for the question image
        images.append(None)
        
        count = 0

        # Examples is a list of (question, image, response) tuples
        for entry in data:
            if count == max:
                break

            # Check language
            if entry["q_lang"] != "en":
                continue

            # Adding examples to conversation
            conversation = list()
            for example in examples:
                conversation.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text":  f"{example[0]}\n"},
                        {"type": "image"}
                    ]
                })
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
                    {"type": "text", "text":  f"{entry["question"]}\n"},
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

            # Insert question image
            image_url = os.path.join(self.image_path, entry["img_name"])
            image = Image.open(image_url)
            images[-1] = image
            true_ans = entry["answer"]
            
            count += 1
            yield prompt, images, true_ans

    def generate_final_prompt(self, output):
        # Instead of generating a new conversation template, alter existing output
        indices = find_all_substr(output, "USER:")
        indices.reverse()
        for i in indices:
            output = output[:(i + 6)] + "<image>" + output[(i + 6):]
        output += f"\n{self.answer_prompt}"

        return output
    
    def run_model(self, prompt, images):
        raw_input = self.processor(
            images=images,
            text=prompt,
            return_tensors='pt',
            padding=True
        ).to(0, torch.float16)

        outputs = list()
        raw_outputs = self.base_model.generate(**raw_input, max_new_tokens=200)
        for raw_output in raw_outputs:
            outputs.append(self.processor.decode(raw_output, skip_special_tokens=True))
        return outputs