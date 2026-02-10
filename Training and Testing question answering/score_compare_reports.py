import pandas as pd
import json
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re

# --- CONFIGURATION ---
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
CSV_FILE_PATH = '/home/jovyan/thesis2-workspace/filtered_dataset_max700_grounded_questions_v2.csv'
JSON_REPORTS_DIR = '/home/jovyan/vscode-workspace/reports'

# --- CORRECTED PROMPT TEMPLATE ---
# The JSON example now uses double curly braces {{...}} to escape them.
PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert radiologist AI assistant. Your task is to analyze two clinical reports and score their semantic similarity based on the most important clinical findings. Focus on significant findings like nodules, effusions, consolidations, and masses. Your output must be ONLY a JSON object and nothing else.<|eot_id|><|start_header_id|>user<|end_header_id|>
Analyze Report A and Report B. Provide a similarity score from 1 (completely different) to 10 (clinically identical).

**REPORT A:**
{report_a}

**REPORT B:**
{report_b}

Respond with ONLY the following JSON object:
{{
  "similarity_score": <score_from_1_to_10_as_a_number>
}}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""


def load_model():
    """Loads the tokenizer and model onto the appropriate device (GPU if available)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    return model, tokenizer

def get_local_llama_score(model, tokenizer, report_a: str, report_b: str) -> float | None:
    """
    Generates a similarity score using the locally loaded Llama 3.1 model.
    """
    prompt = PROMPT_TEMPLATE.format(report_a=report_a, report_b=report_b)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False,
    )
    
    response_text = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            print(f"Warning: No JSON object found in model response: '{response_text}'")
            return None
            
        result_json = json.loads(json_match.group(0))
        score = result_json.get("similarity_score")

        if isinstance(score, (int, float)):
            return float(score)
        else:
            print(f"Warning: Could not find a valid number for 'similarity_score' in response: {result_json}")
            return None

    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error: Could not parse LLM response: {e}. Response was: '{response_text}'")
        return None

def main():
    """
    Main function to load the model, process all reports, and calculate the average score.
    """
    print("Loading Llama 3.1 8B model... (This may take a few minutes)")
    model, tokenizer = load_model()
    print("Model loaded successfully.")

    try:
        df = pd.read_csv(CSV_FILE_PATH)
    except FileNotFoundError:
        print(f"Error: The file {CSV_FILE_PATH} was not found.")
        return

    scores = []
    failed_reports = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Scoring Reports"):
        patient_id = row['Patient ID']
        report_a_text = str(row['Current Findings'])

        json_filename = f'train_{patient_id}_b_1.nii.json'
        json_file_path = os.path.join(JSON_REPORTS_DIR, json_filename)

        if not os.path.exists(json_file_path):
            failed_reports.append(patient_id)
            continue

        try:
            with open(json_file_path, 'r') as f:
                json_data = json.load(f)
                report_b_text = json_data.get('report')
                if not report_b_text:
                    failed_reports.append(patient_id)
                    continue
        except (json.JSONDecodeError, KeyError):
            failed_reports.append(patient_id)
            continue
        
        score = get_local_llama_score(model, tokenizer, report_a_text, report_b_text)
        
        if score is not None:
            scores.append(score)
        else:
            failed_reports.append(patient_id)

    print("\n--- Processing Complete ---")
    if scores:
        average_score = sum(scores) / len(scores)
        print(f"Successfully scored {len(scores)} out of {len(df)} report pairs.")
        print(f"Average Similarity Score: {average_score:.2f}")
    else:
        print("No reports were successfully scored.")

    if failed_reports:
        unique_failed = sorted(list(set(failed_reports)))
        print(f"\nFailed to process {len(unique_failed)} unique reports.")

if __name__ == "__main__":
    main()