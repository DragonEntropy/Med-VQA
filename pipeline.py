from abc import ABC, abstractmethod
import json

class Pipeline(ABC):
    def __init__(self, model, output_file, answer_file, count=-1, batch_size=1):
        self.model = model
        self.output_file = output_file
        self.answer_file = answer_file
        self.batch_size = batch_size
        self.count = count

    @abstractmethod
    def method(self, data, image_path):
        return None
    
    def run(self):
        with open(self.model.data_path, 'r') as data_file:
            data = json.load(data_file)
            self.method(data)


class DirectPipeline(Pipeline):
    def __init__(self, model, output_file, answer_file, count=-1, batch_size=1):
        super().__init__(model, output_file, answer_file, count, batch_size)
        
    def method(self, data):
        initial_prompts = self.model.provide_initial_prompts(data, max=self.count, batch_size=self.batch_size, direct=True)
        for i, (prompt, images, true_answers) in enumerate(initial_prompts):
            print(f"Running batch {i + 1}")

            # Generating CoT responses from initial prompts
            print('Running model for answer step')
            outputs = self.model.run_model(prompt, images)

            for output in outputs:
                self.output_file.write(f"<ENTRY START>\n{output}\n\n")
            for answer in true_answers:
                self.answer_file.write(f"{answer}\n")


class ZeroShotPipeline(Pipeline):
    def __init__(self, model, output_file, answer_file, count=-1, batch_size=1):
        super().__init__(model, output_file, answer_file, count, batch_size)
        
    def method(self, data):
        initial_prompts = self.model.provide_initial_prompts(data, max=self.count, batch_size=self.batch_size)
        blacklist = [31]
        for i, (prompts, images, true_answers) in enumerate(initial_prompts):
            skip = False
            for j in range(i * self.batch_size, (i + 1) * self.batch_size):
                if j in blacklist:
                    skip = True
            if skip:
                continue
            print(f"Running batch {i + 1}")

            # Generating CoT responses from initial prompts
            print('Running model for CoT step')
            outputs = self.model.run_model(prompts, images)
                
            # Creating final prompts
            print("Generating final prompts from CoT outputs")
            final_prompts = self.model.generate_final_prompts(outputs)

            # Generating final answers from prompts
            print("Running model for final answers")
            outputs = self.model.run_model(final_prompts, images)

            for output in outputs:
                self.output_file.write(f"<ENTRY START>\n{output}\n\n")
            for answer in true_answers:
                self.answer_file.write(f"{answer}\n")


class FewShotPipeline(Pipeline):
    def __init__(self, model, output_file, answer_file, count=-1, batch_size=1, example_image=True, k=1):
        super().__init__(model, output_file, answer_file, count, batch_size)
        self.k = k
        self.model.examples = self.model.examples[:min(k, len(self.model.examples))]
        self.include_example_image = example_image
        
    def method(self, data):
        initial_prompts = self.model.provide_initial_prompts(data, max=self.count, batch_size=self.batch_size, example_image=self.include_example_image, examples=self.model.examples)
        blacklist = [31]
        for i, (prompts, images, true_answers) in enumerate(initial_prompts):
            skip = False
            for j in range(i * self.batch_size, (i + 1) * self.batch_size):
                if j in blacklist:
                    skip = True
            if skip:
                continue
            print(f"Running batch {i + 1}")

            # Generating CoT responses from initial prompts
            print('Running model for CoT step')
            print(prompts[0])
            outputs = self.model.run_model(prompts, images)
                
            # Generating final prompts
            print("Generating final prompts from CoT outputs")
            final_prompt = self.model.generate_final_prompts(outputs, example_image=self.include_example_image)
            
            print("Running model for final answers")
            print(final_prompt[0])
            outputs = self.model.run_model(final_prompt, images)

            for output in outputs:
                self.output_file.write(f"<ENTRY START>\n{output}\n\n")
            for answer in true_answers:
                self.answer_file.write(f"{answer}\n")


class AutoShotPipeline(FewShotPipeline):
    def __init__(self, model, output_file, answer_file, count=-1, batch_size=1, example_image=True, k=1):
        super().__init__(model, output_file, answer_file, count, batch_size, example_image, k)
    
        self.model.examples = {
            "Abnormality": [(
                "Where is/are the abnormality located?",
                "xmlab102/source.jpg",
                "Let's think step by step. 1. The image is a CT scan of the chest. 2. The abnormality appears as a dark area in the right lung field. 3. This dark area is not consistent with normal lung tissue and could indicate an abnormality such as a mass, consolidation, or other pathology. 4. The abnormality is located in the right lung field.\nSo, the answer is: The abnormality is located in the right lung field."
            )],
            "Color": [(
                "What color is the lung in the picture?",
                "xmlab378/source.jpg",
                "Let's think step by step. 1. The image is a black-and-white X-ray.2. In X-ray images, the lungs appear as dark areas because they are filled with air, which is less dense than bone and other tissues.3. The dark areas in the image correspond to the lungs.Therefore, the lung in the picture is black.\nSo, the answer is black."
            )],
            "KG": [(
                "Which organs/organ in the picture belong to the respiratory system?",
                "xmlab299/source.jpg",
                "Let's think step by step. The organs/organ in the picture that belong to the respiratory system are:1. **Lungs**: The two large, dark areas on either side of the spine are the lungs. They are responsible for gas exchange, where oxygen is taken in and carbon dioxide is expelled.2. **Trachea**: The central, vertical structure in the middle of the image is the trachea, which is the main airway leading to the lungs.3. **Diaphragm**: The curved, dome-shaped structure at the bottom of the image is the diaphragm, a muscle that plays a crucial role in breathing.These structures are essential components of the respiratory system, which is responsible for the exchange of gases between the body and the environment.\nSo, the answer is: **Lungs, Trachea, Diaphragm**."
            )],
            "Modality": [(
                "What modality is used to take this image?",
                "xmlab102/source.jpg",
                "Let's think step by step. 1. The image is a cross-sectional view of the chest, which is characteristic of a computed tomography (CT) scan. 2. CT scans use X-rays to create detailed images of the body's internal structures. 3. The image shows clear, detailed structures such as bones, muscles, and air-filled spaces, which are typical features of a CT scan.Given these observations, the correct answer is:CT scan.\nSo, the answer is CT scan."
            )],
            "Organ": [(
                "What is the main organ in the image?",
                "xmlab102/source.jpg",
                "Let's think step by step. The image is a cross-sectional view of a human chest, likely from a CT scan. The main organ visible in this image is the heart, located in the center of the image. The heart is surrounded by the lungs, which are visible on either side of the heart. The ribs and other structures are also visible around the heart and lungs.\nSo, the answer is: Heart."
            )],
            "Plane": [(
                "What is the scanning plane of this image?",
                "xmlab219/source.jpg"
                "Let's think step by step. 1. The image shows a cross-sectional view of the chest, with the heart and major blood vessels visible in the center. 2. The ribs are visible on the sides, indicating that the image is a transverse (axial) scan. 3. The vertebral column is also visible at the bottom of the image, further confirming that this is a transverse scan.Therefore, the scanning plane of this image is transverse (axial).\nSo, the answer is transverse (axial)."
            )],
            "Position": [(
                "Where is/are the abnormality located?",
                "xmlab102/source.jpg",
                "Let's think step by step. 1. The image is a CT scan of the chest. 2. The abnormality appears as a dark area in the right lung field. 3. This dark area is not consistent with normal lung tissue and could indicate an abnormality such as a mass, consolidation, or other pathology. 4. The abnormality is located in the right lung field.\nSo, the answer is: The abnormality is located in the right lung field."
            )],
            "Quantity": [(
                "How many kidneys in this image?",
                "xmlab208/source.jpg",
                "Let's think step by step. 1. **Identify the kidneys**: In a CT scan of the abdomen, the kidneys are typically visible as two bean-shaped structures located on either side of the spine.2. **Count the visible structures**: In the image provided, there are two distinct bean-shaped structures on either side of the spine.Therefore, there are **2 kidneys** in this image.\nSo, the answer is **2**."
            )],
            "Shape": [(
                "What is the shape of larynx in this image?",
                "xmlab427/source.jpg",
                "Let's think step by step. 1. The image is a cross-sectional view of the neck, likely from a CT scan.2. The larynx is the part of the neck that contains the vocal cords and is responsible for producing sound.3. In this image, the larynx appears as a dark, oval-shaped structure in the center of the image.4. The shape of the larynx in this image is consistent with the typical appearance of the larynx in a cross-sectional view.Therefore, the shape of the larynx in this image is an oval.\nSo, the answer is oval."
            )],
            "Size": [(
                "What is the largest organ in the picture?",
                "xmlab102/source.jpg",
                "Let's think step by step. The image is a CT scan of the chest. The largest organ in the picture is the lung. The lungs are the two large, spongy organs located in the chest cavity, occupying a significant portion of the thoracic cavity. They are responsible for the exchange of gases between the air we breathe and the bloodstream.\nSo, the answer is: lung."
            )]
        }