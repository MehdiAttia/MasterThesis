import argparse
import json
import os
import ast
import pandas as pd
import evaluate
from tqdm import tqdm
import torch

def main(args):
    """
    Evaluates machine-generated answers against ground truth answers using 
    standard NLP metrics: BLEU, ROUGE, and BERTScore.
    """
    print("--- Starting NLP Metric Evaluation (BLEU, ROUGE, BERTScore) ---")

    # --- Step 1: Load the Metrics ---
    print("--> Loading metrics (BLEU, ROUGE, BERTScore)...")
    try:
        bleu_metric = evaluate.load("bleu")
        rouge_metric = evaluate.load("rouge")
        bertscore_metric = evaluate.load("bertscore")
        print("--> Metrics loaded successfully.")
    except Exception as e:
        print(f"[FATAL ERROR] Could not load metrics. Details: {e}")
        return

    # --- Step 2: Load the ground truth Q&A data from the CSV ---
    print(f"--> Loading ground truth Q&A from: '{args.ground_truth_csv}'")
    try:
        df = pd.read_csv(args.ground_truth_csv)
        ground_truth_map = {}
        
        # Matches the logic of the original script (filtering for 'a' and 'b')
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

    # --- Step 4: Align data ---
    print("--> Aligning generated answers with ground truth answers...")
    
    predictions = []
    references = []
    
    # We also keep a list for BLEU specifically, as it expects references to be a list of lists
    # e.g., references = ["correct"] -> bleu_references = [["correct"]]
    references_for_bleu = []

    for item in generated_data:
        question_text = item['question'].strip()
        generated_answer = item['answer'].strip()
        
        if question_text in ground_truth_map:
            ground_truth_answer = ground_truth_map[question_text]
            
            predictions.append(generated_answer)
            references.append(ground_truth_answer)
            references_for_bleu.append([ground_truth_answer])

    print(f"--> Successfully aligned {len(predictions)} evaluation pairs.")
    if not predictions:
        print("[ERROR] No matching questions found between predictions and ground truth.")
        return

    # --- Step 5: Compute Scores ---
    print("--> Computing scores...")
    
    results = {}

    # 1. BLEU
    # BLEU calculates n-gram overlap.
    print("   ...Computing BLEU")
    bleu_results = bleu_metric.compute(predictions=predictions, references=references_for_bleu)
    results['bleu'] = bleu_results['bleu']

    # 2. ROUGE
    # ROUGE calculates recall (overlap of words/phrases).
    print("   ...Computing ROUGE")
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)
    results['rouge1'] = rouge_results['rouge1']
    results['rouge2'] = rouge_results['rouge2']
    results['rougeL'] = rouge_results['rougeL']

    # 3. BERTScore
    # BERTScore calculates semantic similarity using embeddings.
    # We check for CUDA to speed this up.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   ...Computing BERTScore (using device: {device})")
    
    bert_results = bertscore_metric.compute(
        predictions=predictions, 
        references=references, 
        lang="en", 
        device=device,
        model_type="microsoft/deberta-xlarge-mnli" # You can use 'roberta-large' for faster/smaller inference
    )
    
    # BERTScore returns a list of scores (one per item), so we take the average F1
    avg_bert_f1 = sum(bert_results['f1']) / len(bert_results['f1'])
    results['bertscore_f1'] = avg_bert_f1

    # --- Step 6: Save Results ---
    final_results = {
        "model_name": os.path.basename(args.predictions_file).replace('.json', ''),
        "total_samples_evaluated": len(predictions),
        "scores": {k: round(v, 4) for k, v in results.items()}
    }

    print("\n--- Evaluation Finished ---")
    print(json.dumps(final_results, indent=2))
    
    try:
        with open(args.output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        print(f"\n--> Scores successfully saved to: '{args.output_file}'")
    except Exception as e:
        print(f"\n[ERROR] Could not save scores to file. Details: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated answers using BLEU, ROUGE, and BERTScore.")
    
    parser.add_argument(
        "--predictions-file",
        type=str,
        default="/home/jovyan/thesis2-workspace/CT-CLIP/CT-CHAT/llava/serve/output_test_set_mine.json",
        help="Path to the JSON file containing your model's generated outputs."
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
        default="nlp_metrics_scores_for_answers.json",
        help="Path to save the final scores."
    )
    
    args = parser.parse_args()
    main(args)