from transformers import AutoModelForCausalLM, AutoProcessor, LlavaForConditionalGeneration, Qwen2VLForConditionalGeneration
import torch
import sys
from PIL import Image
import os

from qwen_model import QwenModel
from model import Model
from misc import find_all_substr

import sys
sys.path.insert(0, 'alex/HuatuoGPT-Vision')
from cli import HuatuoChatbot

class HuaTuoModel(Model):
    model_id = "FreedomIntelligence/HuatuoGPT-Vision-7B"
    model_path = "/mnt/HDD3/vri/Documents/alex/model_cache/huatuo/HuatuoGPT-Vision-7B"
    CoT_prompt = "Let's think step by step."
    answer_prompt = "So, the answer"

    def __init__(self, data_path, image_path):
        super().__init__(data_path, image_path)

        self.base_model = HuatuoChatbot(self.model_path)

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

            prompt = entry["question"]
            prompts.append(prompt)
            images.append(os.path.join(self.image_path, entry["img_name"]))
            
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
        return None
    
    def run_model(self, prompts, images):
        outputs = list()
        for prompt, image in zip(prompts, images):
            print(prompt, image)
            outputs.append(self.base_model.inference(prompt, [image]))
        return outputs
"""
class HuaTuoModel(QwenModel):
    model_id = "FreedomIntelligence/HuatuoGPT-Vision-7B"
    CoT_prompt = "Let's think step by step."
    answer_prompt = "So, the answer"

    def __init__(self, data_path, image_path):
        self.data_path = data_path
        self.image_path = image_path

        self.base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir
        ).to(0)
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
"""