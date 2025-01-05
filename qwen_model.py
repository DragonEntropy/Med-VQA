from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
import os
from PIL import Image
from qwen_vl_utils import process_vision_info

from model import Model

class QwenModel(Model):
    model_id = "Qwen/Qwen2-VL-2B"

    def __init__(self, data_path, image_path):
        super().__init__(data_path, image_path)

        self.base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def provide_initial_prompts(self, data, examples=[], batch_size=1, direct=False, max=-1):
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
            for example in examples:
                conversation.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{example[0]}\n"},
                        {"type": "image", "image": f"{os.path.join(self.image_path, example[1])}"}
                    ]
                })
                conversation.append({
                    "role": "system",
                    "content": [
                        {"type": "text", "text": f"{example[2]}\n"},
                    ]
                })

            # Adding required question to conversation
            conversation.append({
                "role": "user",
                "content": [
                    {"type": "text", "text":  f"{entry["question"]}\n"},
                    {"type": "image", "image": f"{os.path.join(self.image_path, entry["img_name"])}"}
                ]
            }),

            # Skip the chain of thought step if directly questioning
            if direct:
                conversation.append({
                    "role": "system",
                    "content": [
                        {"type": "text", "text": ""},
                    ]
                })
            else:
                conversation.append({
                    "role": "system",
                    "content": [
                        {"type": "text", "text": f"{self.CoT_prompt}"},
                    ]
                })

            # Generate prompt via chat template
            print(conversation)
            prompt = self.processor.apply_chat_template(conversation, tokenisation=False, history=[], add_generation_prompt=True)
            print(prompt)
            image_inputs, video_inputs = process_vision_info(conversation)
            prompts.append(prompt)

            images.append(image_inputs)

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

    def generate_final_prompts(self, outputs):
        # Instead of generating a new conversation template, alter existing outputs
        final_prompts = list()
        for output in outputs:
            # output = self.image_injection_method(output)
            final_prompts.append(output)
        return final_prompts

    def run_model(self, prompts, images):
        print(images, prompts)
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