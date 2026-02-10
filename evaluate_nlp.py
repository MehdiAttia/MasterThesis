import argparse
import json
import os
import torch
import evaluate  # Hugging Face Evaluate library
from tqdm import tqdm
import numpy as np

def expand_path(path):
    return os.path.abspath(os.path.expanduser(path))

def main(args):
    # --- Paths ---
    gt_file = expand_path(args.gt_file)
    gen_file = expand_path(args.gen_file)
    output_file = expand_path(args.output_file)

    # --- Load Data ---
    print(f"--> Loading Ground Truth: {gt_file}")
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    # Map scan_name to ground truth text
    gt_dict = {item['scan_name']: item['current'] for item in gt_data if 'current' in item}

    print(f"--> Loading Generated Reports: {gen_file}")
    with open(gen_file, 'r') as f:
        gen_data = json.load(f)

    # --- Initialize Metrics ---
    print("--> Loading Metrics (BLEU, ROUGE, BERTScore)...")
    # Load metrics from Hugging Face 'evaluate' library
    metric_bleu = evaluate.load("bleu")
    metric_rouge = evaluate.load("rouge")
    metric_bertscore = evaluate.load("bertscore")

    # Detect device for BERTScore (runs much faster on GPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--> BERTScore will run on: {device}")

    # --- Prepare Data for Batching ---
    # NLP metrics are faster when computed in batches rather than loop-by-loop
    scan_names = []
    references = []  # Ground Truths
    predictions = [] # Generated Summaries

    print("--> Aligning Data...")
    for item in gen_data:
        scan_name = item['scan_name']
        generated_text = item.get('report', item.get('generated_findings', ''))
        
        # Ensure we have a matching Ground Truth
        if scan_name in gt_dict:
            scan_names.append(scan_name)
            predictions.append(generated_text)
            references.append(gt_dict[scan_name])

    if len(scan_names) == 0:
        print("Error: No matching scan_names found between files.")
        return

    print(f"--> Computing metrics for {len(scan_names)} reports...")

    # --- 1. Compute BERTScore (Semantic Similarity) ---
    # This checks meaning rather than just exact word overlap
    print("   ...Running BERTScore")
    bert_results = metric_bertscore.compute(
        predictions=predictions, 
        references=references, 
        lang="en", 
        device=device,
        batch_size=32  # Adjust based on GPU VRAM
    )
    # bert_results keys: 'precision', 'recall', 'f1' (lists of floats)

    # --- 2. Compute ROUGE (Recall/Overlap) ---
    # Common for summarization tasks
    print("   ...Running ROUGE")
    rouge_results = metric_rouge.compute(
        predictions=predictions, 
        references=references, 
        use_aggregator=False # False returns a list of scores per item
    )
    # rouge_results keys: 'rouge1', 'rouge2', 'rougeL' (lists of floats)

    # --- 3. Compute BLEU (Precision) ---
    # BLEU expects references to be a list of lists: [[ref1], [ref2]]
    # We calculate per item here to keep granular data
    print("   ...Running BLEU")
    bleu_scores = []
    for pred, ref in tqdm(zip(predictions, references), total=len(predictions), desc="BLEU"):
        # BLEU is harsh on single sentences, works best on longer text
        res = metric_bleu.compute(predictions=[pred], references=[[ref]])
        bleu_scores.append(res['bleu'])

    # --- Compile Results ---
    detailed_results = []
    
    # Accumulators for averages
    sums = {
        "bleu": 0,
        "rouge1": 0,
        "rouge2": 0,
        "rougeL": 0,
        "bert_f1": 0
    }

    print("--> compiling individual scores...")
    for i, scan_name in enumerate(scan_names):
        
        # Extract individual scores
        b_score = bleu_scores[i]
        r1 = rouge_results['rouge1'][i]
        r2 = rouge_results['rouge2'][i]
        rl = rouge_results['rougeL'][i]
        bert_f1 = bert_results['f1'][i]

        # Add to totals
        sums["bleu"] += b_score
        sums["rouge1"] += r1
        sums["rouge2"] += r2
        sums["rougeL"] += rl
        sums["bert_f1"] += bert_f1

        # Create entry
        entry = {
            "scan_name": scan_name,
            "metrics": {
                "BLEU": round(b_score, 4),
                "ROUGE-1": round(r1, 4),
                "ROUGE-2": round(r2, 4),
                "ROUGE-L": round(rl, 4),
                "BERTScore_F1": round(bert_f1, 4)
            },
            "generated_report": predictions[i],
            "ground_truth_snippet": references[i][:100] + "..." # Save space
        }
        detailed_results.append(entry)

    # --- Calculate Averages ---
    count = len(scan_names)
    averages = {k: round(v / count, 4) for k, v in sums.items()}

    print("\n--> Evaluation Complete.")
    print(f"Average BLEU:      {averages['bleu']}")
    print(f"Average ROUGE-1:   {averages['rouge1']}")
    print(f"Average BERTScore: {averages['bert_f1']}")

    final_output = {
        "summary": {
            "total_scans": count,
            "averages": averages
        },
        "details": detailed_results
    }

    # --- Save ---
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=4)
    print(f"--> Results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--gt-file", type=str, default="processed_scan_data_cleaned.json")
    parser.add_argument("--gen-file", type=str, default="final_radiology_reports.json")
    parser.add_argument("--output-file", type=str, default="evaluation_metrics_nlp.json")
    # Note: No model-path needed for standard metrics (BERTScore downloads its own small model automatically)
    
    args = parser.parse_args()
    main(args)