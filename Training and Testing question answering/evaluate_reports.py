import evaluate
import json
import argparse
import os
from datasets import load_dataset
from tqdm import tqdm

from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

def main(args):
    """
    This script evaluates generated radiology reports against a ground truth dataset
    using BLEU, ROUGE, METEOR, and a direct implementation of CIDEr.
    """
    print("--- Starting Model Evaluation ---")

    print("--> Loading evaluation metrics (BLEU, ROUGE, METEOR)...")
    try:
        bleu_metric = evaluate.load('bleu')
        rouge_metric = evaluate.load('rouge')
        meteor_metric = evaluate.load('meteor')
        print("--> Metrics loaded successfully.")
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not load a metric. Details: {e}")
        return

    print(f"--> Loading and filtering ground truth dataset: '{args.dataset_name}'")
    try:
        ds = load_dataset(args.dataset_name, "reports", split="validation")
        filtered_ds = ds.filter(lambda example: example['VolumeName'].endswith('_1.nii.gz'))
        
        ground_truths = {}
        for row in filtered_ds:
            base_filename = os.path.basename(row['VolumeName']).replace('.nii.gz', '')
            full_report = f"Findings: {row['Findings_EN']} Impression: {row['Impressions_EN']}"
            ground_truths[base_filename] = full_report
            
        print(f"--> Found {len(ground_truths)} matching ground truth reports.")
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not load or process the Hugging Face dataset. Details: {e}")
        return

    print(f"--> Loading generated predictions from: '{args.predictions_file}'")
    try:
        with open(args.predictions_file, 'r') as f:
            generated_data = json.load(f)
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not read the predictions JSON file. Details: {e}")
        return

    print("--> Aligning predictions with ground truth references...")
    predictions = []
    references = []
    gts_for_cider = {}
    res_for_cider = {}

    for i, item in enumerate(tqdm(generated_data, desc="Aligning Data")):
        base_filename = item['image'].replace('.npz', '')

        if base_filename in ground_truths:
            if item['conversations_out']:
                prediction_text = item['conversations_out'][0]['answer'].replace("<|eot_id|>", "").strip()
                reference_text = ground_truths[base_filename]

                # Data for BLEU, ROUGE, METEOR (list of strings)
                predictions.append(prediction_text)
                references.append([reference_text])
                
                # --- START: THE DEFINITIVE FIX ---
                # Data for CIDEr (list of dictionaries with a 'caption' key)
                gts_for_cider[i] = [{"caption": reference_text}]
                res_for_cider[i] = [{"caption": prediction_text}]
                # --- END: THE DEFINITIVE FIX ---

    if not predictions:
        print("\n[FATAL ERROR] After alignment, 0 matching predictions were found. Cannot evaluate.")
        return
        
    print(f"--> Successfully aligned {len(predictions)} prediction/reference pairs.")

    print("\n--- Computing Scores ---")
    
    bleu_results = bleu_metric.compute(predictions=predictions, references=references, max_order=4)
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)
    meteor_results = meteor_metric.compute(predictions=predictions, references=references)

    print("--> Manually calculating CIDEr score...")
    tokenizer = PTBTokenizer()
    gts_tokenized = tokenizer.tokenize(gts_for_cider)
    res_tokenized = tokenizer.tokenize(res_for_cider)
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(gts_tokenized, res_tokenized)
    print("--> CIDEr calculation complete.")
    
    final_scores = {
        "model_name": os.path.basename(args.predictions_file).replace('.json', ''),
        "num_samples": len(predictions),
        "bleu_1": bleu_results['precisions'][0],
        "rouge_l": rouge_results['rougeL'],
        "meteor": meteor_results['meteor'],
        "cider": cider_score
    }

    print("\n--- Final Scores ---")
    print(json.dumps(final_scores, indent=2))

    try:
        with open(args.output_file, 'w') as f:
            json.dump(final_scores, f, indent=2)
        print(f"\n--> Scores successfully saved to: '{args.output_file}'")
    except Exception as e:
        print(f"\n[ERROR] Could not save scores to file. Details: {e}")

    print("\n--- Evaluation Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated radiology reports.")
    parser.add_argument("--predictions-file", type=str, default="output_validation_llama.json")
    parser.add_argument("--output-file", type=str, default="evaluation_results.json")
    parser.add_argument("--dataset-name", type=str, default="ibrahimhamamci/CT-RATE")
    args = parser.parse_args()
    main(args)