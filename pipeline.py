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
            for j in range(i * self.batch_size, (i + 1) * self.batch_size):
                if j in blacklist:
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
            for j in range(i * self.batch_size, (i + 1) * self.batch_size):
                if j in blacklist:
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