from abc import ABC, abstractmethod
import json

class Pipeline(ABC):
    def __init__(self, model, output_file, answer_file):
        self.model = model
        self.output_file = output_file
        self.answer_file = answer_file

    @abstractmethod
    def method(self, data, image_path):
        return None
    
    def run(self):
        with open(self.model.data_path, 'r') as data_file:
            data = json.load(data_file)
            self.method(data)


class DirectPipeline(Pipeline):
    def __init__(self, model, output_file, answer_file):
        super().__init__(model, output_file, answer_file)
        
    def method(self, data):
        initial_prompts = self.model.provide_initial_prompts(data, direct=True)
        for i, (prompt, images, true_answer) in enumerate(initial_prompts):
            print(f"Running batch {i + 1}")

            # Generating CoT responses from initial prompts
            print('Running model for answer step')
            outputs = self.model.run_model(prompt, images)

            for output in outputs:
                self.output_file.write(f"<ENTRY START>\n{output}\n\n")
            self.answer_file.write(f"{true_answer}\n")


class ZeroShotPipeline(Pipeline):
    def __init__(self, model, output_file, answer_file):
        super().__init__(model, output_file, answer_file)
        
    def method(self, data):
        initial_prompts = self.model.provide_initial_prompts(data)
        for i, (prompt, images, true_answer) in enumerate(initial_prompts):
            print(f"Running batch {i + 1}")

            # Generating CoT responses from initial prompts
            print('Running model for CoT step')
            outputs = self.model.run_model(prompt, images)
                
            # Generating final prompts
            print("Generating final prompts from CoT outputs")
            final_prompt = self.model.generate_final_prompt(outputs[0])

            
            print("Running model for final answers")
            outputs = self.model.run_model(final_prompt, images)

            for output in outputs:
                self.output_file.write(f"<ENTRY START>\n{output}\n\n")
            self.answer_file.write(f"{true_answer}\n")


class FewShotPipeline(Pipeline):
    def __init__(self, model, output_file, answer_file, k=1):
        super().__init__(model, output_file, answer_file)
        self.k = k
        self.model.examples = self.model.examples[:max(k, len(self.model.examples))]
        
    def method(self, data):
        initial_prompts = self.model.provide_initial_prompts(data, examples=self.model.examples)
        for i, (prompt, images, true_answer) in enumerate(initial_prompts):
            print(f"Running batch {i + 1}")

            # Generating CoT responses from initial prompts
            print('Running model for CoT step')
            outputs = self.model.run_model(prompt, images)
                
            # Generating final prompts
            print("Generating final prompts from CoT outputs")
            final_prompt = self.model.generate_final_prompt(outputs[0])
            
            print("Running model for final answers")
            outputs = self.model.run_model(final_prompt, images)

            for output in outputs:
                self.output_file.write(f"<ENTRY START>\n{output}\n\n")
            self.answer_file.write(f"{true_answer}\n")