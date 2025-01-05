from abc import ABC, abstractmethod
import json

class Pipeline(ABC):
    def __init__(self, model, output_file, answer_file, batch_size=1):
        self.model = model
        self.output_file = output_file
        self.answer_file = answer_file
        self.batch_size = batch_size

    @abstractmethod
    def method(self, data, image_path):
        return None
    
    def run(self):
        with open(self.model.data_path, 'r') as data_file:
            data = json.load(data_file)
            self.method(data)


class DirectPipeline(Pipeline):
    def __init__(self, model, output_file, answer_file, batch_size=1):
        super().__init__(model, output_file, answer_file, batch_size)
        
    def method(self, data):
        initial_prompts = self.model.provide_initial_prompts(data, batch_size=self.batch_size, direct=True)
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
    def __init__(self, model, output_file, answer_file, batch_size=1):
        super().__init__(model, output_file, answer_file, batch_size)
        
    def method(self, data):
        initial_prompts = self.model.provide_initial_prompts(data, batch_size=self.batch_size)
        for i, (prompts, images, true_answers) in enumerate(initial_prompts):
            print(f"Running batch {i + 1}")

            # Generating CoT responses from initial prompts
            print('Running model for CoT step')
            print(prompts[0])
            outputs = self.model.run_model(prompts, images)
                
            # Creating final prompts
            print("Generating final prompts from CoT outputs")
            final_prompts = self.model.generate_final_prompts(outputs)

            # Generating final answers from prompts
            print("Running model for final answers")
            print(final_prompts[0])
            outputs = self.model.run_model(final_prompts, images)

            for output in outputs:
                self.output_file.write(f"<ENTRY START>\n{output}\n\n")
            for answer in true_answers:
                self.answer_file.write(f"{answer}\n")


class FewShotPipeline(Pipeline):
    def __init__(self, model, output_file, answer_file, batch_size=1, k=1):
        super().__init__(model, output_file, answer_file, batch_size)
        self.k = k
        self.model.examples = self.model.examples[:min(k, len(self.model.examples))]
        
    def method(self, data):
        initial_prompts = self.model.provide_initial_prompts(data,  batch_size=self.batch_size, examples=self.model.examples)
        for i, (prompt, images, true_answers) in enumerate(initial_prompts):
            print(f"Running batch {i + 1}")

            # Generating CoT responses from initial prompts
            print('Running model for CoT step')
            print(prompt[0])
            outputs = self.model.run_model(prompt, images)
                
            # Generating final prompts
            print("Generating final prompts from CoT outputs")
            final_prompt = self.model.generate_final_prompts(outputs)
            
            print("Running model for final answers")
            print(final_prompt[0])
            outputs = self.model.run_model(final_prompt, images)

            for output in outputs:
                self.output_file.write(f"<ENTRY START>\n{output}\n\n")
            for answer in true_answers:
                self.answer_file.write(f"{answer}\n")