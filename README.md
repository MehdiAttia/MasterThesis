# Radiology QA + Report Generation Pipeline

This repository contains the end-to-end workflow used in the thesis to:
- Generate clinically grounded questions from prior CT reports
- Prepare data for multimodal QA with precomputed image embeddings
- Run inference with a text LLM + LoRA adapter conditioned on image embeddings
- Summarize Q&A pairs into structured radiology reports
- Evaluate reports and answers using standard NLP metrics and LLM-based judging

The pipeline can be run in modular steps. Below is a practical run order with file-by-file descriptions, inputs/outputs, and example commands.

## Prerequisites
- Python 3.10+
- GPU recommended for model inference and BERTScore
- Key packages: `torch`, `transformers`, `peft`, `tqdm`, `evaluate`, `datasets`, `wandb`, `numpy`, `pandas`, `pycocoevalcap`
- Precomputed CT embeddings (.npz → projector → .pt) and local model checkpoints/paths used in scripts

Note: Several scripts include Linux-style default paths (e.g., `~/thesis2-workspace/...`). Override these with your local paths via CLI args where available.

## CT-CLIP and CT-CHAT Setup
- Purpose: These external repositories provide the multimodal components (encoders, projectors, data formats) used by scripts here. Several paths in the scripts reference CT-CLIP and CT-CHAT directories.
- Get the repos: Search on GitHub for their official repositories (CT-CLIP and CT-CHAT) and follow their README instructions.
- Recommended local layout (adjust as desired):

```powershell
# Clone the repos (replace owner/repo with the official ones you find)
# CT-CLIP
cd $HOME\thesis2-workspace
git clone https://github.com/ibrahimethemhamamci/CT-CLIP.git

# CT-CHAT
git clone https://github.com/ibrahimethemhamamci/CT-CHAT.git
```

- Install dependencies (run in each repo directory):

```powershell
# In CT-CLIP
cd $HOME\thesis2-workspace\CT-CLIP
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Optional (editable installs per CT-CLIP README)
cd transformer_maskgit
pip install -e .
cd ..\CT_CLIP
pip install -e .

# In CT-CHAT
cd $HOME\thesis2-workspace\CT-CHAT
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

- Notes:
	- Some components may require CUDA-enabled PyTorch; verify versions in the repos’ READMEs.
	- If the repos use additional setup scripts (e.g., downloading models or datasets), run those steps before using this pipeline.
	- Update the paths in the scripts here (e.g., `--model-base`, `--model-path`, embeddings directories) to point to your local CT-CLIP/CT-CHAT folders on Windows.
	- References:
		- CT-CLIP: https://github.com/ibrahimethemhamamci/CT-CLIP
		- CT-CHAT: https://github.com/ibrahimethemhamamci/CT-CHAT
		- CT-RATE dataset: https://huggingface.co/datasets/ibrahimhamamci/CT-RATE

## High-Level Workflow
1. Question generation from prior findings CSV
2. Clean and transform questions into LLaVA-style JSON
3. Project raw image embeddings to LLM space for SFT/inference
4. (Optional) Supervised Fine-Tuning (LoRA) over multimodal inputs
5. Inference to answer questions per image
6. Report generation from Q&A pairs
7. Evaluation of answers and reports

## Run Order (Suggested)
- Step 0: Prepare your inputs (CSV of prior/current reports, embeddings directories, model paths).

1) Generate grounded clinical questions (CSV → JSON)
- Purpose: Create per-scan question lists grounded in prior findings.
- Script: `Question Generation/question_generator.py`
- Input: CSV with columns `Patient ID`, `Prior Findings`, `Current Findings`
- Output: `processed_scan_data_qwen.json`
- Usage: Edit `INPUT_CSV_PATH` and `MODEL_ID` at the top, then run:
```bash
python "Question Generation/question_generator.py"
```

2) Clean question lists
- Purpose: Keep only valid questions ending with `?`.
- Script: `clean_questions.py`
- Input: `processed_scan_data.json` or `processed_scan_data_qwen.json` (rename if needed)
- Output: `processed_scan_data_cleaned.json`
- Usage:
```bash
python clean_questions.py
```

3) Create LLaVA-style validation JSON (per-question entries)
- Option A (from cleaned JSON): `prepare_merged.py` → `merged_output_valid.json`
- Option B (directly from CSV): `Training and Testing question answering/create_flattened_json.py`
- Usage (A):
```bash
python prepare_merged.py
```
- Usage (B):
```bash
python "Training and Testing question answering/create_flattened_json.py" --input-csv <path/to/filtered_dataset.csv> --output-json merged_output_valid.json
```

4) Pre-process embeddings for SFT/inference (projector)
- Purpose: Convert 512-d `.npz` features into 4096-d `.pt` tensors via the model’s multimodal projector.
- Script: `Training and Testing question answering/preprocess_for_sft.py`
- Inputs: `--source-embeddings-dir`, `--model-path`, `--model-base`
- Output dir: `--dest-embeddings-dir`
- Usage:
```bash
python "Training and Testing question answering/preprocess_for_sft.py" --model-path <llava_checkpoint_dir> --model-base meta-llama/Meta-Llama-3.1-8B-Instruct --source-embeddings-dir <path/to/npz> --dest-embeddings-dir <path/to/pt>
```

5) (Optional) Supervised Fine-Tuning (LoRA)
- Purpose: Train LoRA adapters on multimodal inputs.
- Script: `Training and Testing question answering/grid.py`
- Inputs: `--data-path` (finetune JSON with `user`/`assistant` text), `--sft-embeddings-dir`
- Output: Adapter checkpoints under `--output-dir`
- Usage:
```bash
python "Training and Testing question answering/grid.py" --model-base meta-llama/Meta-Llama-3.1-8B-Instruct --data-path <finetune_data.json> --sft-embeddings-dir <path/to/pt> --output-dir <out_dir>
```

6) Inference: Answer questions per image
- Purpose: Generate answers conditioned on image embeddings.
- Option A (validation JSON): `Training and Testing question answering/small_ctchat_validation_llama.py`
- Option B (test set JSONL): `Training and Testing question answering/run_test_set.py`
- Option C (top-level): `question_answering.py`
- Outputs: Q&A results (e.g., `output_SFT_finetuned_results_augmented.json`, `output_test_set_their.json`)
- Usage (A):
```bash
python "Training and Testing question answering/small_ctchat_validation_llama.py" --model-base <llama_base> --model-path <lora_checkpoint> --embeddings-dir <path/to/pt> --output-file output_SFT_finetuned_results_augmented.json
```
- Usage (B):
```bash
python "Training and Testing question answering/run_test_set.py" --input-file <test_set.json> --model-base <llama_base> --model-path <lora_checkpoint> --embeddings-dir <path/to/pt> --output-file output_test_set_their.json
```
- Usage (C):
```bash
python question_answering.py --model-base <llama_base> --model-path <lora_checkpoint> --embeddings-dir <path/to/pt> --output-file output_SFT_finetuned_results_augmented.json
```

7) Report generation from Q&A
- Purpose: Summarize Q&A into structured Findings/Report text.
- Option A (concise Findings): `report_generator.py` → `final_radiology_reports.json`
- Option B (structured sections + per-file outputs): `Training and Testing question answering/generate_ct_reports.py` → Markdown + JSON files
- Usage (A):
```bash
python report_generator.py --input-file output_SFT_finetuned_results_augmented.json --output-file final_radiology_reports.json --model-path <text_model_path>
```
- Usage (B):
```bash
python "Training and Testing question answering/generate_ct_reports.py" --input <qa_results.json> --model-path <text_model_path> --output-dir <reports_dir>
```

8) Evaluation
- Reports vs Ground Truth (NLP metrics): `evaluate_nlp.py`
```bash
python evaluate_nlp.py --gt-file processed_scan_data_cleaned.json --gen-file final_radiology_reports.json --output-file evaluation_metrics_nlp.json
```
- Reports via LLM Judging (JSON score): `evaluate_reports_total.py`
```bash
python evaluate_reports_total.py --gt-file processed_scan_data_cleaned.json --gen-file final_radiology_reports.json --output-file evaluation_scores_cot.json --model-path <llama_base>
```
- Alternative report eval (HF CT-RATE + BLEU/ROUGE/METEOR/CIDEr): `Training and Testing question answering/evaluate_reports.py`
```bash
python "Training and Testing question answering/evaluate_reports.py" --predictions-file <output_validation_llama.json> --output-file evaluation_results.json --dataset-name ibrahimhamamci/CT-RATE
```
- Answers vs Ground Truth (NLP metrics): `Training and Testing question answering/evaluate_answers_nlp.py`
```bash
python "Training and Testing question answering/evaluate_answers_nlp.py" --predictions-file <answers.json> --ground-truth-csv <filtered_dataset.csv> --output-file nlp_metrics_scores_for_answers.json
```
- Answers via LLM Judging: `Training and Testing question answering/score_answers_with_llama.py`
```bash
python "Training and Testing question answering/score_answers_with_llama.py" --model-base <llama_base> --predictions-file <answers.json> --ground-truth-csv <filtered_dataset.csv> --output-file llama_answer_scores.json
```
- Compare ground truth vs generated reports via LLM: `Training and Testing question answering/score_compare_reports.py`
```bash
python "Training and Testing question answering/score_compare_reports.py"
```

## File-by-File Summary
- `Question Generation/question_generator.py`: Generates 3–5 grounded clinical questions per scan from prior findings (CSV → JSON). Edit constants at top.
- `clean_questions.py`: Filters any non-question strings from processed JSON.
- `prepare_merged.py`: Transforms cleaned JSON into per-question LLaVA-style entries with placeholder answers.
- `Training and Testing question answering/create_flattened_json.py`: Flattens CSV Q&A into LLaVA-style JSON for validation (a→b changes only).
- `Training and Testing question answering/generate_report_json.py`: Builds LLaVA-style JSON prompts for report generation for all `_1.nii.gz` scans in a directory.
- `Training and Testing question answering/preprocess_for_sft.py`: Projects 512-d `.npz` features to 4096-d `.pt` via mm_projector.
- `Training and Testing question answering/grid.py`: Runs LoRA SFT over multimodal inputs; logs to Weights & Biases.
- `Training and Testing question answering/small_ctchat_validation_llama.py`: Inference over validation JSON; merges image embeddings into prompt via special token.
- `Training and Testing question answering/ctchat_validation_llama.py`: Inference using LLaVA builder; projects 512-d features to 4096-d via `mm_projector` and streams outputs; saves `output_TRAINING_SET_inference.json`.
- `Training and Testing question answering/run_test_set.py`: Inference over test JSONL; filters out augmented images.
- `question_answering.py`: Top-level inference script similar to validation/test, saving `output_SFT_finetuned_results_augmented.json`.
- `report_generator.py`: Summarizes Q&A into Findings using a text-only LLM.
- `Training and Testing question answering/generate_ct_reports.py`: Produces full structured reports (Exam/Comparison/Technique/Findings/Impression).
- `evaluate_nlp.py`: BLEU/ROUGE/BERTScore evaluation per report vs ground truth.
- `evaluate_reports_total.py`: LLM-based similarity scoring with strict JSON output.
- `evaluate_reports.py`: Evaluates generated reports against CT-RATE validation using BLEU, ROUGE, METEOR, and manual CIDEr; aligns predictions to filenames ending with `_1.nii.gz`.
- `Training and Testing question answering/evaluate_reports.py`: Alternative evaluation using HF dataset CT-RATE and CIDEr.
- `Training and Testing question answering/evaluate_answers_nlp.py`: NLP metrics for answers.
- `Training and Testing question answering/score_answers_with_llama.py`: LLM judge for answers.
- `Training and Testing question answering/score_compare_reports.py`: LLM judge comparing ground truth vs generated reports.

## Tips & Notes
- Paths: Many defaults point to Linux home directories. Override via CLI flags or edit constants.
- Embeddings: Ensure `.pt` embeddings match filenames (e.g., `train_<id>_b_1.nii.gz` → `train_<id>_b_1.pt`).
- Tokens: Some scripts use `[IMG]` or `<image>` as special tokens; keep consistent with tokenizer setup.
