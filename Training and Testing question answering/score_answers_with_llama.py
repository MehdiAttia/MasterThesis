import argparse
import json
import os
import re
import torch
import transformers
import pandas as pd
import ast
from tqdm import tqdm

def parse_score_from_text(text: str) -> int | None:
    """
    Parses the LLM's output to find a similarity score from 1 to 10.
    It's designed to be robust against minor formatting variations.
    """
    # Look for the ideal format: "Score: X" or "Similarity Score: X"
    match = re.search(r"score:\s*(\d{1,2})", text, re.IGNORECASE)
    if match:
        try:
            score = int(match.group(1))
            if 0 <= score <= 10:
                return score
        except (ValueError, IndexError):
            pass

    # If that fails, look for any standalone number from 0 to 10
    match = re.search(r"\b([0-9]|10)\b", text)
    if match:
        try:
            score = int(match.group(1))
            return score
        except (ValueError, IndexError):
            pass

    return None

def main(args):
    """
    Uses Llama 3.1 8B to judge the clinical similarity between ground truth answers
    and machine-generated answers for a set of questions about CT scans.
    """
    print("--- Starting LLM-based Answer Evaluation ---")

    # --- Step 1: Load the Llama 3.1 8B judging model ---
    args.model_base = os.path.expanduser(args.model_base)
    print(f"--> Loading judging LLM from local path: '{args.model_base}'")
    if not torch.cuda.is_available():
        print("[FATAL ERROR] This script requires a GPU.")
        return
        
    try:
        judging_pipeline = transformers.pipeline(
            "text-generation",
            model=args.model_base,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
        print("--> Judging LLM loaded successfully.")
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not load the Llama 3.1 model. Is the path correct?")
        print(f"Details: {e}")
        return

    # --- Step 2: Load the ground truth Q&A data from the CSV ---
    print(f"--> Loading ground truth Q&A from: '{args.ground_truth_csv}'")
    try:
        df = pd.read_csv(args.ground_truth_csv)
        # Create a dictionary for fast lookup: { "question_text": "ground_truth_answer_text" }
        ground_truth_map = {}
        for _, row in df.iterrows():
            if row['Prior Scan ID'] == 'a' and row['Current Scan ID'] == 'b':
                questions = ast.literal_eval(row['Questions'])
                answers = ast.literal_eval(row['Answers'])
                for q, a in zip(questions, answers):
                    ground_truth_map[q.strip()] = a.strip()
        print(f"--> Loaded {len(ground_truth_map)} ground truth Q&A pairs.")
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not load or process the ground truth CSV. Details: {e}")
        return

    # --- Step 3: Load the generated predictions ---
    print(f"--> Loading generated predictions from: '{args.predictions_file}'")
    try:
        with open(args.predictions_file, 'r') as f:
            generated_data = json.load(f)
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not read the predictions JSON file. Details: {e}")
        return

    # --- Step 4: Align data and prepare for evaluation ---
    print("--> Aligning generated answers with ground truth answers...")
    evaluation_triplets = []
    for item in generated_data:
        question_text = item['question'].strip()
        generated_answer = item['answer'].strip()
        
        if question_text in ground_truth_map:
            ground_truth_answer = ground_truth_map[question_text]
            evaluation_triplets.append({
                'question': question_text,
                'generated_answer': generated_answer,
                'ground_truth_answer': ground_truth_answer
            })
    
    print(f"--> Successfully aligned {len(evaluation_triplets)} evaluation triplets.")
    if not evaluation_triplets:
        return

    # --- Step 5: Loop through triplets, get scores from LLM, and calculate average ---
    valid_scores = []
    failed_parses = 0
    
    system_prompt = "You are an expert radiologist. Your task is to evaluate the clinical accuracy and similarity between a ground truth answer and a machine-generated answer, in the context of a specific question about a Chest CT scan. Focus purely on the factual and clinical correctness of the generated answer compared to the ground truth."

    for triplet in tqdm(evaluation_triplets, desc="Evaluating with LLM"):
        user_prompt = f"""
[CONTEXT QUESTION]:
{triplet['question']}

[GROUND TRUTH ANSWER]:
{triplet['ground_truth_answer']}

[GENERATED ANSWER]:
{triplet['generated_answer']}

[INSTRUCTION]:
On a scale of 0 to 10, where 0 is completely incorrect/irrelevant and 10 is perfectly accurate and equivalent in clinical meaning to the ground truth, how would you score the generated answer? Provide only the numerical score in the format: 'Score: X/10'.
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        # Run inference
        outputs = judging_pipeline(
            messages,
            max_new_tokens=20,
            do_sample=False,
            temperature=0.0
        )
        
        generated_text = outputs[0]["generated_text"][-1]["content"]
        
        # Parse the score
        score = parse_score_from_text(generated_text)
        
        if score is not None:
            valid_scores.append(score)
        else:
            failed_parses += 1
            tqdm.write(f"Warning: Could not parse score from LLM output: '{generated_text}'")

    # --- Step 6: Calculate and display the final average score ---
    if not valid_scores:
        print("\n[ERROR] No valid scores could be parsed from the LLM's responses.")
        return

    average_score = sum(valid_scores) / len(valid_scores)
    
    final_results = {
        "model_name": os.path.basename(args.predictions_file).replace('.json', ''),
        "llm_judge": os.path.basename(args.model_base),
        "average_answer_similarity_score": round(average_score, 4),
        "total_samples_evaluated": len(valid_scores),
        "failed_parses": failed_parses
    }
    
    print("\n--- LLM Evaluation Finished ---")
    print(json.dumps(final_results, indent=2))
    
    try:
        with open(args.output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\n--> Scores successfully saved to: '{args.output_file}'")
    except Exception as e:
        print(f"\n[ERROR] Could not save scores to file. Details: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use an LLM to evaluate the similarity of generated answers.")
    
    parser.add_argument(
        "--model-base",
        type=str,
        default="~/thesis2-workspace/CT-CLIP/models/Meta-Llama-3.1-8B-Instruct",
        help="Path to the Llama 3.1 8B Instruct model used as the judge."
    )
    parser.add_argument(
        "--predictions-file",
        type=str,
        required=True,
        help="Path to the JSON file containing your model's generated outputs (e.g., output_TRAINING_SET_inference.json)."
    )
    parser.add_argument(
        "--ground-truth-csv",
        type=str,
        default="/home/jovyan/thesis2-workspace/filtered_dataset_max700_grounded_questions_v2.csv",
        help="Path to the original CSV file containing the ground truth questions and answers."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="llama_answer_scores_with_augment_on_test_mine.json",
        help="Path to save the final LLM-based evaluation scores."
    )
    
    args = parser.parse_args()
    main(args)