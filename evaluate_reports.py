import argparse
import torch
import json
import os
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))

def extract_score(text):
    """
    Extracts the score from the model's response (e.g., "Score: 9").
    """
    match = re.search(r'(?:Score|Rating|Result):\s*(\d+)', text, re.IGNORECASE)
    if match:
        return max(1, min(10, int(match.group(1))))
    
    # Fallback: find last number
    numbers = re.findall(r'\b(10|[0-9])\b', text)
    if numbers:
        return int(numbers[-1])
    return 0

def create_precision_prompt(ground_truth, generated):
    """
    Prompts the model to check for ACCURACY only, ignoring omissions.
    """
    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a Medical Fact-Checker. Your task is to verify the accuracy of a 'Generated Summary' against a 'Ground Truth Report'.\n\n"
        "### RULES:\n"
        "1. **Focus on ACCURACY, not COMPLETENESS.**\n"
        "   - Do NOT penalize the Generated Summary for leaving out information.\n"
        "   - The Generated Summary is expected to be shorter than the Ground Truth.\n"
        "2. **Check for HALLUCINATIONS or CONTRADICTIONS.**\n"
        "   - If the Summary says 'X is normal' and Ground Truth says 'X is normal', that is perfect.\n"
        "   - If the Summary says 'Mass detected' but Ground Truth says 'No mass', that is a major error.\n"
        "3. **Ignore Minor Style Differences.**\n"
        "   - 'Pleural effusion seen' vs 'Effusion noted in pleura' is a match.\n\n"
        "### SCORING (1-10):\n"
        "10: Every statement in the Generated Summary is supported by the Ground Truth.\n"
        "8-9: Statements are mostly correct, but maybe a slight inaccuracy in measurement or location.\n"
        "5-7: The Summary makes a claim that contradicts the Ground Truth (e.g., wrong side, wrong organ).\n"
        "1-4: The Summary is hallucinating findings that do not exist at all.\n\n"
        "Format:\n"
        "Analysis: [Briefly check the claims]\n"
        "Score: [Number]<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "### Ground Truth Report:\n"
        f"{ground_truth}\n\n"
        "### Generated Summary (to verify):\n"
        f"{generated}\n\n"
        "Verify the claims. Output your Analysis and Score.<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt

def main(args):
    # --- Paths ---
    gt_file = expand_path(args.gt_file)
    gen_file = expand_path(args.gen_file)
    output_file = expand_path(args.output_file)
    model_path = expand_path(args.model_path)

    # --- Load Data ---
    print(f"--> Loading Ground Truth: {gt_file}")
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    gt_dict = {item['scan_name']: item['current'] for item in gt_data if 'current' in item}

    print(f"--> Loading Generated Reports: {gen_file}")
    with open(gen_file, 'r') as f:
        gen_data = json.load(f)

    # --- Load Model ---
    print(f"--> Loading Evaluator: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    scores = []
    total_score = 0
    count = 0

    print("--> Starting Fact-Checking...")
    
    for item in tqdm(gen_data, desc="Evaluating"):
        scan_name = item['scan_name']
        generated_text = item.get('report', item.get('generated_findings', ''))
        
        if scan_name not in gt_dict:
            continue

        gt_text = gt_dict[scan_name]

        # 1. Create Prompt
        prompt = create_precision_prompt(gt_text, generated_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # 2. Generate
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200, 
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # 3. Decode
        response = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        score = extract_score(response)

        # 4. Save
        result_entry = {
            "scan_name": scan_name,
            "similarity_score": score,
            "analysis": response,
            "generated_report": generated_text
        }
        scores.append(result_entry)
        
        if score > 0:
            total_score += score
            count += 1
            # Debug print
            if count <= 2:
                print(f"\n[DEBUG] {scan_name} -> Score: {score}")
                print(f"Analysis: {response[:100]}...")

    # --- Summary ---
    avg_score = (total_score / count) if count > 0 else 0
    print(f"\n--> Evaluation Complete.")
    print(f"--> Average Accuracy Score: {avg_score:.2f} / 10")

    final_output = {
        "summary": {
            "total_scans": count,
            "average_accuracy": avg_score
        },
        "details": scores
    }

    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=4)
    print(f"--> Results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--gt-file", type=str, default="processed_scan_data_cleaned.json")
    parser.add_argument("--gen-file", type=str, default="final_radiology_reports.json")
    parser.add_argument("--output-file", type=str, default="evaluation_scores_cot.json")
    parser.add_argument("--model-path", type=str, default="~/thesis2-workspace/CT-CLIP/models/Meta-Llama-3.1-8B-Instruct")
    
    args = parser.parse_args()
    main(args)