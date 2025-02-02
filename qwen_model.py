from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, AutoTokenizer
import torch
import os
from qwen_vl_utils import process_vision_info
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from model import Model

class QwenModel(Model):
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    CoT_prompt = "Let's think step by step."
    answer_prompt = "So, the answer"

    def __init__(self, data_path, image_path):
        super().__init__(data_path, image_path)

        self.base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            cache_dir=self.cache_dir,
            attn_implementation="eager"
        )

        min_pixels = 256*28*28
        max_pixels = 1024*28*28 
        self.processor = AutoProcessor.from_pretrained(self.model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        self.tokeniser = AutoTokenizer.from_pretrained(self.model_id)
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
            if entry["q_lang"] != "en": # or entry["content_type"] != "KG":
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
        print(count)
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
        print(len(prompts[0]))
        
        raw_outputs = self.base_model.generate(**raw_input, max_new_tokens=200, output_attentions=store_attention, return_dict_in_generate=True)
        for raw_output in raw_outputs["sequences"]:
            outputs.append(self.processor.decode(raw_output, skip_special_tokens=True))

        if store_attention:
            attentions = raw_outputs["attentions"]  # Nested list of attention scores
            tokenized_input = raw_input["input_ids"]  # Tokenized input

            # Determine token indices for text
            text_token_mask = (tokenized_input != self.processor.tokenizer.pad_token_id) & \
                            (tokenized_input != self.processor.image_token)  # Adjust for image tokens
            text_token_indices = text_token_mask.nonzero(as_tuple=True)[1]  # Extract text token indices

            layer_index = 0
            batch_index = 0
            head_index = 0
            other_index = 0

            # Extract attention matrix for the specific layer and head
            attention_matrix = attentions[layer_index][batch_index][head_index].detach().cpu().numpy()

            # Extract text-to-text attention
            text_to_text_attention = attention_matrix[np.ix_(text_token_indices.cpu(), text_token_indices.cpu())]

            # Visualize text-to-text attention
            plt.figure(figsize=(10, 8))
            sns.heatmap(text_to_text_attention, cmap="viridis")
            plt.title(f"Text-to-Text Attention (Layer {layer_index}, Head {head_index})")
            plt.xlabel("Text Tokens")
            plt.ylabel("Text Tokens")
            plt.savefig("alex/results/test.png")
        if store_attention:
            with open("alex/results/raw_output.txt", "w") as dump:
                dump.write(str(raw_outputs["attentions"]))

            for attention in attentions:
                print(len(attention))
                for at in attention:
                    print(at.shape)
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
