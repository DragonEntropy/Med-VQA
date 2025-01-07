import os
from abc import ABC, abstractmethod

class Model(ABC):
    cache_dir = os.path.join(os.getcwd(), "alex", "model_cache")

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
So, the answer is a CT scan\n"""
        )]

    @abstractmethod
    def provide_initial_prompts(self, data, examples=[], batch_size=1, direct=False, example_image=True, max=-1):
        yield None
    
    @abstractmethod
    def generate_final_prompts(self, outputs, example_image=True):
        return None
    
    @abstractmethod
    def run_model(self, prompts, images):
        return None