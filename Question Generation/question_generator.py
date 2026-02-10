import json
import csv
import re
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
MODEL_ID = "Qwen/Qwen1.5-1.8B" #"meta-llama/Meta-Llama-3.1-8B-Instruct"
INPUT_CSV_PATH = "/home/jovyan/thesis2-workspace/filtered_dataset_max700_grounded_questions_v2.csv"  # <--- REPLACE THIS WITH YOUR CSV PATH
OUTPUT_FILENAME = "processed_scan_data_qwen.json"

# --- PROMPTS ---
RADIOLOGIST_SYSTEM_PROMPT = """
You are an expert radiologist AI assistant. Your task is to meticulously review a 'Prior Scan Findings' report.
Based *only* on the information presented in this prior report, generate a concise list of 3-5 pertinent clinical questions.
These questions should focus on aspects of the prior report that a radiologist would deem important to track, reassess, or look for changes in a *subsequent* follow-up scan or report.
The questions must be directly grounded in findings or statements from the provided 'Prior Scan Findings'. Do not introduce information not present in the prior report.
Format your output as a numbered list of questions.
"""

RADIOLOGIST_USER_PROMPT_TEMPLATE = """
Prior Scan Findings:
\"\"\"
{prior_findings_text}
\"\"\"

Based on the above findings, generate a list of 3-5 pertinent clinical questions a radiologist would formulate anticipating a follow-up examination:
"""

def extract_clean_text(text):
    if not isinstance(text, str):
        return ""
    return text.strip()

def parse_model_output(raw_output):
    """Cleans the output into a Python list of question strings."""
    lines = raw_output.split('\n')
    questions = []
    for line in lines:
        # Clean up numbering (e.g., "1. ", "2) ", "- ", "Questions:")
        cleaned_line = re.sub(r"^\s*(Questions:|List of Questions:|Generated Questions:)\s*", "", line, flags=re.IGNORECASE).strip()
        cleaned_line = re.sub(r"^\s*[\d\)\.\-\s]+\s*", "", cleaned_line).strip()
        if cleaned_line:
            questions.append(cleaned_line)
    return questions

def main():
    if not os.path.exists(INPUT_CSV_PATH):
        print(f"Error: CSV file not found at {INPUT_CSV_PATH}")
        return

    print(f"Loading Tokenizer: {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading Model: {MODEL_ID} (using Hugging Face)...")
    try:
        # 'device_map="auto"' will automatically assign layers to GPU/CPU
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
    except Exception as e:
        print(f"Error loading model with Hugging Face: {e}")
        print("Ensure you have 'autoawq' installed: pip install autoawq")
        return

    all_data_records = []

    print(f"Reading data from {INPUT_CSV_PATH}...")
    with open(INPUT_CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            patient_id = row['Patient ID']
            prior_text = extract_clean_text(row['Prior Findings'])
            current_text = extract_clean_text(row['Current Findings'])
            
            # Format Scan Name: train_{ID}_b_1.nii.gz
            scan_name = f"train_{patient_id}_b_1.nii.gz"

            print(f"Generating questions for {scan_name} (Patient {patient_id})...")

            # Construct Messages
            messages = [
                {"role": "system", "content": RADIOLOGIST_SYSTEM_PROMPT},
                {"role": "user", "content": RADIOLOGIST_USER_PROMPT_TEMPLATE.format(prior_findings_text=prior_text)}
            ]
            
            # Tokenize Prompt
            input_ids = tokenizer.apply_chat_template(
                messages, 
                tokenize=True, 
                add_generation_prompt=True, 
                return_tensors="pt"
            ).to(model.device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids,
                    max_new_tokens=350,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )

            # Decode Output
            # We slice [input_ids.shape[1]:] to remove the input prompt from the result
            generated_text = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
            
            # Parse Questions
            questions_list = parse_model_output(generated_text)
            
            record = {
                "scan_name": scan_name,
                "prior": prior_text,
                "current": current_text,
                "questions": questions_list
            }
            
            all_data_records.append(record)

    # Save to JSON
    print(f"Saving {len(all_data_records)} records to {OUTPUT_FILENAME}...")
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(all_data_records, f, indent=4, ensure_ascii=False)
        
    print("Processing complete.")

if __name__ == "__main__":
    main()