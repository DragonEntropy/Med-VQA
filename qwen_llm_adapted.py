# Adapted from Evelyn Doan's code: kdoa8440@uni.sydney.edu.au

import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import seaborn as sns
import matplotlib.pyplot as plt

from misc import get_collection_dim, find_all_substr

GLOBAL_COUNT = 0

# Set cache directories
os.environ["TRANSFORMERS_CACHE"] = "/mnt/HDD3/cache/huggingface"
os.environ["HF_HOME"] = "/mnt/HDD3/cache/huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["DISABLE_FLASH_ATTN"] = "1"
os.environ["FLASH_ATTENTION_FORCE_EAGER"] = "1"

def gpu_allocate():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected, using CPU.")
    return device

def load_model_and_tokenizer(model_name, device):
    cache_dir = os.path.join(os.getcwd(), "alex", "model_cache")
    """
    Load the model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", trust_remote_code=True, cache_dir=cache_dir)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, cache_dir=cache_dir, attn_implementation="eager").to(device)
    print(model.config)
    torch.cuda.empty_cache()
    return model, tokenizer

def plot_heatmap(outputs, target_index, tokens):
    global GLOBAL_COUNT

    ref = torch.stack(outputs["attentions"][-1]).shape
    attention = torch.zeros([ref[4] + 1, ref[4]]).to(0)
    print(attention.shape)
    for i, at in enumerate(outputs["attentions"]):
        att = torch.stack(at).to(0).mean(dim=0).mean(dim=0).mean(dim=0)
        if i == 0:
            padding_y = torch.zeros([att.shape[0], ref[4] - att.shape[1]]).to(0)
            padding_x = torch.zeros([ref[4] + 1 - att.shape[0], ref[4]]).to(0)
            padded_at = torch.cat([torch.cat([att, padding_y], dim=-1), padding_x], dim=-2)
        else:
            padding_y = torch.zeros([att.shape[0], ref[4] - att.shape[1]]).to(0)
            padding_x_left = torch.zeros([att.shape[1] - att.shape[0], ref[4]]).to(0)
            padding_x_right = torch.zeros([ref[4] + 1 - att.shape[1], ref[4]]).to(0)
            padded_at = torch.cat([torch.cat([padding_x_left, torch.cat([att, padding_y], dim=-1)], dim=-2), padding_x_right], dim=-2)
        attention += padded_at
  
    attention_matrix = attention.cpu().numpy()
    plt.figure(figsize=(150, 150))
    sns.heatmap(attention_matrix, xticklabels=tokens, yticklabels=tokens, cmap="viridis")
    plt.title("Attention Weights")

    plt.savefig(f"alex/results/test_{GLOBAL_COUNT}.png")
    GLOBAL_COUNT += 1
    plt.cla()
    plt.clf()

    """
    attention = torch.stack(outputs["attentions"][target_index]).to(torch.float32)
    
    attention_matrix = attention.mean(dim=0).mean(dim=0).mean(dim=0).cpu().numpy()
    # print(attention_matrix.shape)

    plt.figure(figsize=(150, 150))
    sns.heatmap(attention_matrix.transpose(), xticklabels=[0], yticklabels=tokens, cmap="viridis")
    plt.title("Attention Weights")

    plt.savefig(f"alex/results/test_{GLOBAL_COUNT}.png")
    GLOBAL_COUNT += 1
    plt.cla()
    plt.clf()
    """

def load_data():
    # Load dataset
    dataset = load_dataset("VodLM/medqa", name="tw")
    data = [{"question": item["question"], "answers": item["answers"], "target": item["target"]} for item in
            dataset["validation"]]
    return data

def find_answer_index(token_list):
    token_list.reverse()
    for i, token in enumerate(token_list):
        if token.isnumeric():
            token_list.reverse()
            return -(i + 1)
    return None

def generate_stage1_prompts(batch, tokenizer):
    """
    Generate prompts for Stage 1: Rationale Generation.
    """
    prompts = []
    questions = []
    choices = []
    true_answers = []

    # Few-shot examples for reasoning
    few_shot_examples = [
        {
            "question": "Four weeks after starting hydrochlorothiazide, a 49-year-old man with hypertension comes to the physician because of muscle cramps and weakness. His home medications also include amlodipine. His blood pressure today is 176/87 mm Hg. Physical examination shows no abnormalities. The precordial leads of a 12-lead ECG are shown. The addition of which of the following is most likely to have prevented this patient’s condition?",
            "choices": ["Torsemide", "Nifedipine", "Eplerenone", "Hydralazine"],
            "rationale": "Let’s think step by step. The patient has started hydrochlorothiazide. He now presents with muscle cramps and weakness and an ECG that supports the diagnosis of hypokalemia. The addition of (0) Torsemide is a loop diuretic and would likely aggravate the hypokalemia. (1) Nifedipine is a calcium antagonist and would not alleviate the hypokalemia. (2) Eplerenone is a potassium-sparing diuretic and would likely decrease the chance of hypokalemia. (3) Hydralazine is a potent vasodilator and would not decrease the risk of hypokalemia.",
            "answer": "2"
        },
        {
            "question": "A 57-year-old woman comes to the emergency department because of severe pain around her right eye, blurred vision in the same eye, and a headache for the past 4 hours. She is nauseous but has not vomited. She can see colored bright circles when she looks at a light source. She is currently being treated for a urinary tract infection with trimethoprim-sulfamethoxazole. She appears uncomfortable. Vital signs are within normal limits. Examination shows visual acuity of 20/20 in the left eye and counting fingers at 5 feet in the right eye. The right eye shows conjunctival injection and edema of the cornea. The right pupil is dilated and fixed. Intravenous analgesia and antiemetics are administered. Which of the following is the most appropriate next step in management?",
            "choices": ["Perform ultrasound biomicroscopy", "Perform gonioscopy", "Perform fundoscopy",
                        "Administer topical steroids"],
            "rationale": "Let’s think step by step. The patient has severe pain, blurred vision, and a headache. She is also nauseous but has not vomited. She can see bright colored circles when she looks at a light source. The right eye shows conjunctival injection and edema of the cornea. The right pupil is dilated and fixed. The tentative (most likely) diagnosis, given the patient’s symptoms, is acute primary angle-closure glaucoma. Analgesics have been administered. The next step is to confirm the tentative diagnosis using the gold-standard test. (0) Ultrasound biomicroscopy could be used but are not widely available, (1) Gonioscopy is the gold-standard test to confirm the diagnosis of acute primary angle-closure glaucoma, (2) Fundoscopy is not as reliable as gonioscopy, (3) Topical steroids is not relevant; the possible medical therapy would be timolol, apraclonidine or pilocarpine. The most appropriate next step in management is to perform gonioscopy.",
            "answer": "1"
        },
        {
            "question": "Which of the following causes, resulting in type II diabetes, is most likely not a major causative mechanism?",
            "choices": ["Of insulin resistance", "Insulin secretion", "Inadequate secretion of glucagon",
                        "Excessive free fatty acids"],
            "rationale": "Let’s think step by step. (0) Insulin resistance is a core mechanism in type II diabetes, as it reduces the body’s ability to respond to insulin, leading to hyperglycemia. This is a major cause and not the correct answer. (1) Impaired insulin secretion from pancreatic beta cells is central to the progression of type II diabetes, as it prevents proper regulation of blood glucose. This is also a major cause and not correct. (2) Inadequate secretion of glucagon does not significantly contribute to type II diabetes. While glucagon dysregulation can affect glucose metabolism, inadequate glucagon secretion is not a primary mechanism in the disease, making this the correct answer. (3) Excessive free fatty acids promote insulin resistance and impair beta-cell function, both critical in the development of type II diabetes. Thus, this is not the correct answer.",
            "answer": "2"
        },
        {
            "question": "Pregabalin produces its analgesic effect by targeting which of the following ion channels?",
            "choices": ["sodium channel", "calcium channel", "chloride channel", "potassium channel"],
            "rationale": "Let's think step by step. (0) Pregabalin does not act on sodium channels. While sodium channels are crucial for action potential propagation, they are not the primary target of pregabalin. Thus, this is not the correct answer. (1) Pregabalin binds specifically to the α₂-δ subunit of voltage-gated calcium channels. This action reduces calcium influx in neurons, leading to decreased release of excitatory neurotransmitters like glutamate and substance P, which are involved in pain signaling. This is the primary mechanism of its analgesic effect, making this the correct answer. (2) Chloride channels are associated with inhibitory neurotransmission, primarily through GABAergic pathways. Pregabalin does not interact with chloride channels for its effect, so this option is incorrect. (3) Potassium channels regulate neuronal excitability, but pregabalin does not primarily target these channels. Therefore, this option is not correct.",
            "answer": "2"
        },
        {
            "question": "When using the immunofluorescence method to detect anti-nuclear antibodies (ANA), different patterns of nuclear staining can be observed. Which of the following patterns is most associated with systemic sclerosis?",
            "choices": ["Centromere pattern", "Peripheral pattern", "Cytoplasmic pattern", "Homogenous pattern"],
            "rationale": "Let's think step by step. (0) The centromere pattern is strongly associated with limited systemic sclerosis (also known as CREST syndrome). This is the correct answer. (1) The peripheral pattern is often seen in conditions like systemic lupus erythematosus (SLE) and is not primarily associated with systemic sclerosis. Therefore, this is not the correct answer. (2) The cytoplasmic pattern indicates antibodies against cytoplasmic components and is typically associated with certain autoimmune diseases but not specifically systemic sclerosis. Thus, this is not correct. (3) The homogenous pattern is usually observed in diseases like SLE or drug-induced lupus and is not predominantly linked to systemic sclerosis. Hence, this is not the correct answer.",
            "answer": "0"
        },
        {
            "question": "Gartner duct cyst is:",
            "choices": ["Skene duct origin", "Müllerian duct origin", "Mesonephric origin", "Bartholin duct origin"],
            "rationale": "Let's think step by step. Gartner duct cysts originate from remnants of (2) the mesonephric duct (also known as the Wolffian duct). These remnants can persist in females and sometimes form cysts along the lateral walls of the vagina. (0) the Skene duct, associated with paraurethral glands near the female urethra, is not involved in the formation of Gartner duct cysts. Similarly, (1) the Müllerian duct, which develops into the uterus, fallopian tubes, and the upper third of the vagina, is also unrelated to Gartner duct cysts. Finally, (3) the Bartholin duct, associated with glands located in the vulvar region, is not embryologically connected to Gartner duct cysts.",
            "answer": "2"
        },
        {
            "question": "As shown in retinal photographs, the patients most likely diagnosis of why?",
            "choices": ["Acute central retinal artery occlusion", "Retinopathy of premature children",
                        "Background diabetic retinopathy (background diabetic retinopathy; BDR)",
                        "Proliferative diabetic retinopathy (proliferative diabetic retinopathy; PDR)"],
            "rationale": "Let's think step by step. (0) Acute central retinal artery occlusion presents with sudden, painless vision loss and a cherry-red spot at the macula, which is not consistent with the described findings. Therefore, this option is incorrect. (1) Retinopathy of premature children occurs in premature infants due to abnormal retinal vascular development and is unlikely in the context of an adult patient. Therefore, this option is incorrect. (2) Background diabetic retinopathy (BDR) s characterized by microaneurysms, dot and blot hemorrhages, and hard exudates, matching the retinal findings described in the case. Therefore, this option is correct. (3) Proliferative diabetic retinopathy (PDR) is an advanced stage of diabetic retinopathy with neovascularization, which is more severe than the findings associated with the case. Therefore, this option is incorrect.",
            "answer": "2"
        },
        {
            "question": "The law governing medical practitioners excludes which of the following groups from its definition of a physician?",
            "choices": ["Chinese medicine practitioners", "Veterinary", "Physician", "Dentists"],
            "rationale": "Let's think step by step. (0) Chinese medicine practitioners are typically categorized under alternative or complementary medicine in most jurisdictions. They are not commonly included in the standard definition of a physician. Therefore, this is not the correct answer. (1) Veterinarians are trained specifically to diagnose and treat animals and are governed by veterinary-specific laws. They are excluded from the definition of physicians in medical practice laws. Therefore, this is the correct answer. (2) Physicians are central to the scope of medical practice laws, as they provide direct care to humans. They are explicitly included in the definition of medical practitioners. Therefore, this option is incorrect. (3) Dentists are often regulated separately under dental practice laws but may occasionally fall under broader medical practitioner definitions. However, they are not excluded as physicians in this context. So, this option is not the correct answer.",
            "answer": "1"
        }
    ]
    few_shot_examples = []

    # Prepare the few-shot examples section
    few_shot_examples_text = ""
    for example in few_shot_examples:
        few_shot_examples_text += (
            f"Question: {example['question']}\n"
            f"Options:\n"
            f"- 0) {example['choices'][0]}\n"
            f"- 1) {example['choices'][1]}\n"
            f"- 2) {example['choices'][2]}\n"
            f"- 3) {example['choices'][3]}\n"
            f"Reasoning: {example['rationale']}\n"
            f"Conclusion: Therefore, the final answer is: {example['answer']}\n\n"
        )

    for entry in batch:
        question = entry["question"]
        choice = entry["answers"]
        correct_answer = entry["target"]

        # Use Qwen's chat template for structuring input
        messages = [
            {"role": "system", "content": "You are a highly knowledgeable medical expert trained to make logical decisions about medical diagnoses and treatments. Your task is to answer multiple-choice questions based on the given context."},
            {"role": "user", "content": f"Question: {question}\n"
                                        f"Options:\n"
                                        f"- 0) {choice[0]}\n"
                                        f"- 1) {choice[1]}\n"
                                        f"- 2) {choice[2]}\n"
                                        f"- 3) {choice[3]}\n"
                                        "Let's reason through the options logically and identify the best answer."}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)
        questions.append(question)
        choices.append(choice)
        true_answers.append(correct_answer)

    return prompts, questions, choices, true_answers


# Generate prompts for Stage 2: Answer Extraction
def generate_stage2_prompts(rationales, batch, tokenizer):
    """
    Generate prompts for Stage 2: Final Answer Extraction.
    """
    prompts = []
    for i, entry in enumerate(batch):
        question = entry["question"]
        choice = entry["answers"]

        # Construct the Qwen chat-style prompt for Stage 2
        messages = [
            {"role": "system", "content": "You are a highly knowledgeable medical expert trained to make logical decisions about medical diagnoses and treatments."},
            {"role": "user", "content": f"{rationales[i]}\n"
                                        f"Therefore, the final answer for the question in singular number (0, 1, 2, or 3) is: "}
        ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)

    return prompts

# Perform inference with the model
def model_inference(model, tokenizer, prompts, device, max_new_tokens=100, store_attention=False):
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=1024).to(device)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # Ensure inputs contain valid key-value pairs for model.generate
    outputs = model.generate(
        **inputs,  # Pass input_ids explicitly
        do_sample=True,
        max_new_tokens=max_new_tokens,
        temperature=0.1,
        top_p=0.3,
        top_k=10,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
        output_attentions=store_attention,
        return_dict_in_generate=True
    )
    torch.cuda.empty_cache()
    
    global GLOBAL_COUNT

    if store_attention:
        get_collection_dim(outputs["sequences"])
        for i, output in enumerate(outputs["sequences"]):
            tokens = tokenizer.convert_ids_to_tokens(output)
            index = find_answer_index(tokens)
            print(index, tokens[-10:])
            if -index > len(outputs["attentions"]):
                GLOBAL_COUNT += 1
                continue
            
            plot_heatmap(outputs, index, tokens)


    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs["sequences"]]


# Extract numeric answers from the response
def extract_answer(response):
    print(response)
    # pattern = r"Therefore, the final answer for the question in singular number \(0, 1, 2, or 3\) is:\s*(\d)"
    pattern = r"(\d)"
    match = re.findall(pattern, response)
    if match:
        # print(match)
        return int(match[-1])
    return -1

# Full pipeline implementation
def few_shot_pipeline(model, tokenizer, data, device, batch_size=10):
    results = []
    total_correct = 0
    total_processed = 0

    total_batches = (len(data) + batch_size - 1) // batch_size
    # total_batches = 3
    for batch_idx in range(total_batches):
        if batch_idx >= 10:
            continue
        # print(f"Processing batch {batch_idx + 1}/{total_batches}...")
        torch.cuda.empty_cache()

        batch = data[batch_idx * batch_size: (batch_idx + 1) * batch_size]

        stage1_prompts, questions, choices, correct_answers = generate_stage1_prompts(batch, tokenizer)
        rationales = model_inference(model, tokenizer, stage1_prompts, device, max_new_tokens=512, store_attention=True)

        # Clear memory after stage 1
        del stage1_prompts
        torch.cuda.empty_cache()

        stage2_prompts = generate_stage2_prompts(rationales, batch, tokenizer)
        final_answers = model_inference(model, tokenizer, stage2_prompts, device, max_new_tokens=10, store_attention=True)

        # Clear memory after stage 2
        del stage2_prompts, rationales
        torch.cuda.empty_cache()

        # Evaluate predictions
        for i, response in enumerate(final_answers):
            predicted_answer = extract_answer(response)
            is_correct = predicted_answer == int(correct_answers[i])
            total_correct += is_correct
            total_processed += 1

            results.append({
                "question": questions[i],
                "choices": choices[i],
                "correct_answer": int(correct_answers[i]),
                "predicted_answer": predicted_answer,
                "is_correct": is_correct,
                "response": response,
            })

            # Log responses
            """
            print(f"Question: {questions[i]}")
            print(f"Model Response: {response}")
            print(f"Predicted Answer: {predicted_answer}, Correct Answer: {correct_answers[i]}")
            print("-" * 50)
            """
            

    overall_accuracy = (total_correct / total_processed) * 100
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")
    return results, overall_accuracy

def main():
    data = load_data()
    device = gpu_allocate()
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    model, tokenizer = load_model_and_tokenizer(model_name, device)
    result, accuracy = few_shot_pipeline(model, tokenizer, data, device, 1)

if __name__ == "__main__":
    main()