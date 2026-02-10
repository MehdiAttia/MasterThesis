import argparse
import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def group_data_by_image(data):
    """
    Groups the flat list of Q&A pairs into a dictionary keyed by image filename.
    """
    grouped = {}
    for entry in data:
        img = entry['image']
        if img not in grouped:
            grouped[img] = []
        grouped[img].append({
            "question": entry['question'],
            "answer": entry['answer']
        })
    return grouped

def create_prompt(qa_list):
    """
    Creates the prompt for the LLM to summarize the Q&A into a Findings section.
    """
    # formatted_qa = "\n".join([f"Q: {item['question']}\nA: {item['answer']}" for item in qa_list])
    
    # Using a cleaner format for the model to read
    formatted_qa = ""
    for item in qa_list:
        formatted_qa += f"- Finding: {item['answer']} (Context: {item['question']})\n"

    prompt = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a radiologist assistant. Your task is to write the 'Findings' section of a CT scan report.\n"
        "Use ONLY the provided facts. Do not add outside information. Do not write an 'Impression' or 'Conclusion'.\n"
        "Combine the facts into a fluent, professional medical paragraph.<|eot_id|>\n"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "Here are the specific findings identified in the scan:\n"
        f"{formatted_qa}\n"
        "Write the Findings paragraph now.<|eot_id|>\n"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return prompt

def main(args):
    # --- File Paths ---
    input_path = os.path.abspath(args.input_file)
    output_path = os.path.abspath(args.output_file)
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    # --- Load Data ---
    print(f"--> Loading Q&A results from: {input_path}")
    with open(input_path, 'r') as f:
        raw_data = json.load(f)
    
    grouped_data = group_data_by_image(raw_data)
    print(f"--> Found {len(grouped_data)} unique scans to process.")

    # --- Load Model (Text Only) ---
    print(f"--> Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    final_results = []

    print("--> Generating Reports...")
    for image_name, qa_pairs in tqdm(grouped_data.items(), desc="Processing Scans"):
        
        prompt = create_prompt(qa_pairs)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False, # Deterministic for reporting
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and extract only the new text
        generated_text = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Clean up the text (strip whitespace)
        report_text = generated_text.strip()

        # Append to results
        final_results.append({
            "scan_name": image_name,
            "qa_pairs": qa_pairs,
            "generated_findings": report_text
        })

    # --- Save Output ---
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=4)
    
    print(f"\n--> Done! Reports saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Use the output file from your PREVIOUS step as the input here
    parser.add_argument("--input-file", type=str, default="output_SFT_finetuned_results_augmented.json")
    parser.add_argument("--output-file", type=str, default="final_radiology_reports.json")
    # You can use the base Llama 3 model path here
    parser.add_argument("--model-path", type=str, default="/home/jovyan/thesis2-workspace/CT-CLIP/models/Meta-Llama-3.1-8B-Instruct")
    args = parser.parse_args()
    main(args)