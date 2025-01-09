from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
import os
from qwen_vl_utils import process_vision_info

from model import Model

class QwenModel(Model):
    model_id = "Qwen/Qwen2-VL-7B-Instruct"
    CoT_prompt = "Let's think step by step."
    answer_prompt = "So, the answer"

    def __init__(self, data_path, image_path):
        super().__init__(data_path, image_path)

        self.base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.examples = [("Which part of the body does this image belong to?",
            "xmlab102/source.jpg",
            "Let's think step by step. The image is a CT scan of the chest. The CT scan shows the lungs, heart, and other structures of the chest. 1. The lungs are visible in the center of the image.2. The heart is located in the center of the chest, slightly to the right.3. The other structures visible in the image are the ribs, spine, and other bones of the chest.Based on these observations, the image belongs to the chest.\nSo, the answer is chest.\n"
        )]

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
            for example in examples:
                conversation.append({
                    "role": "user",
                    "content": [
                        {"type": "image", "image": f"file://{os.path.join(self.image_path, example[1])}"},
                        {"type": "text", "text": f"{example[0]}\n"}
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
                    {"type": "image", "image": f"file://{os.path.join(self.image_path, entry["img_name"])}"},
                    {"type": "text", "text":  f"{entry["question"]}\n"}
                ]
            })

            # Skip the chain of thought step if directly questioning
            if not direct:
                conversation.append({
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"{self.CoT_prompt}"},
                    ]
                })

            # Generate prompt via chat template
            prompt = self.processor.apply_chat_template(conversation, tokenisation=False, history=[], add_generation_prompt=True)

            # Force model to extend existing prompt by removing trailing characters
            if not direct:
                prompt = prompt[:-(len("<|im_end|>\n<|im_start|>assistant") + 1)]
            
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

    def generate_final_prompts(self, outputs, example_image=True):
        # Instead of generating a new conversation template, alter existing outputs
        final_prompts = list()
        for output in outputs:
            output = self.image_injection_method(output)
            final_prompts.append(output)
        return final_prompts

    def run_model(self, prompts, images):
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
        output = ""
        entities = ["user", "system", "assistant"]
        last_entity = None
        for line in prompt.split("\n"):
            if line in entities:
                if last_entity:
                    output += "<|im_end|>\n"
                output += f"<|im_start|>{line}\n"
                last_entity = line
            elif last_entity == "user":
                output += f"<|vision_start|><|image_pad|><|vision_end|>{line}\n"
                last_entity = "expired"
            else:
                output += line
        output += f"\n{self.answer_prompt}"

        return output
