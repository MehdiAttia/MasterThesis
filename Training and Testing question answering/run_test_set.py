import argparse
import torch
import json
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm

def main(args):
    # --- Path handling ---
    args.model_base = os.path.abspath(os.path.expanduser(args.model_base))
    args.model_path = os.path.abspath(os.path.expanduser(args.model_path))
    args.embeddings_dir = os.path.abspath(os.path.expanduser(args.embeddings_dir))
    
    # Handle input file path
    if not os.path.isabs(args.input_file):
        # If relative path, assume it's relative to the script location
        args.input_file = os.path.join(os.path.dirname(__file__), args.input_file)
    args.input_file = os.path.abspath(os.path.expanduser(args.input_file))

    print(f"--> Loading TEXT-ONLY base model from: {args.model_base}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({'additional_special_tokens': ['[IMG]']})

    model = AutoModelForCausalLM.from_pretrained(
        args.model_base,
        device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))

    print(f"--> Loading fine-tuned LoRA adapter from: {args.model_path}")
    model = PeftModel.from_pretrained(model, args.model_path)
    model = model.merge_and_unload()
    print("--> Model + adapter merged successfully.")

    # --- Load Input JSON (JSONL format) ---
    data_to_process = []
    
    try:
        print(f"--> Reading input file: {args.input_file}")
        with open(args.input_file, 'r') as file:
            for line in file:
                if line.strip():
                    item = json.loads(line)
                    
                    # --- FILTERING STEP ---
                    # Check if 'image_file' exists, otherwise try 'image'
                    img_key = item.get("image_file", item.get("image"))
                    
                    if img_key and "_aug" not in img_key:
                        data_to_process.append(item)
                        
    except FileNotFoundError:
        print(f"[FATAL] Missing input file: {args.input_file}")
        return

    print(f"\n--- Processing {len(data_to_process)} NORMAL samples (Augmented images filtered out) ---")

    # --- Load embeddings ---
    print("--> Pre-loading image embeddings...")
    image_embeddings = {}
    
    for element in tqdm(data_to_process, desc="Loading embeddings"):
        # Handle key difference if necessary
        image_key = element.get("image_file", element.get("image"))
        
        # Determine .pt filename
        if image_key.endswith(".nii.gz"):
            image_file = image_key.replace(".nii.gz", ".pt")
        else:
            image_file = image_key

        image_path = os.path.join(args.embeddings_dir, image_file)

        if os.path.exists(image_path):
            emb = torch.load(image_path, map_location=model.device)
            emb = emb.to(dtype=next(model.parameters()).dtype)
            image_embeddings[image_key] = emb
        else:
            if image_key not in image_embeddings:
                print(f"[WARNING] Missing embedding for {image_key} at {image_path}")

    print(f"--> {len(image_embeddings)} embeddings loaded onto {model.device}")

    # --- Inference loop ---
    output_save = []
    img_token_id = tokenizer.convert_tokens_to_ids('[IMG]')
    pad_token_id = tokenizer.pad_token_id

    print("\nRunning inference...")
    for element in tqdm(data_to_process, desc="Inference"):
        img_filename = element.get("image_file", element.get("image"))
        
        # Double check strictly ensuring no augs slip through
        if "_aug" in img_filename:
            continue

        if img_filename not in image_embeddings:
            continue

        img_emb = image_embeddings[img_filename]

        # Extract Question from text
        # Format: "user\n<image>\nQUESTION\nassistant\n..."
        full_text = element["text"]
        try:
            prompt_part = full_text.split("assistant\n")[0]
            question = prompt_part.split("<image>\n")[1].strip()
        except IndexError:
            print(f"[SKIP] Format error in text: {full_text[:50]}...")
            continue

        # Create Prompt
        prompt = f"user\n[IMG]\n{question}\nassistant\n"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        try:
            img_token_index = torch.where(input_ids == img_token_id)[1][0]
        except IndexError:
            print(f"[SKIP] No [IMG] token found in prompt for {img_filename}")
            continue

        input_embed_layer = model.get_input_embeddings()
        embeds_before = input_embed_layer(input_ids[:, :img_token_index])
        embeds_after = input_embed_layer(input_ids[:, img_token_index + 1:])
        final_input_embeds = torch.cat([embeds_before, img_emb, embeds_after], dim=1)

        attention_mask = torch.ones(final_input_embeds.shape[:2], dtype=torch.long, device=model.device)

        with torch.inference_mode():
            output_ids = model.generate(
                inputs_embeds=final_input_embeds,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=pad_token_id
            )

        decoded_full = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        answer = decoded_full.replace(prompt, "").strip()

        output_save.append({
            "image": img_filename,
            "question": question,
            "answer": answer
        })

    # --- Save output ---
    output_file_path = os.path.join(os.path.dirname(__file__), args.output_file)
    with open(output_file_path, "w") as f:
        json.dump(output_save, f, indent=4)
    print(f"\n--> Inference complete. Results saved to {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # --- New Argument added here ---
    parser.add_argument("--input-file", type=str, default="/home/jovyan/sft_adapter_5_aug/test_set.json", 
                        help="Path to the input JSONL test file")
    
    parser.add_argument("--model-base", type=str, default="~/thesis2-workspace/CT-CLIP/models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--model-path", type=str, default="/home/jovyan/thesis2-workspace/CT-CLIP/models/llama_3.1_8b/models/CT-CHAT/llava-lora-llama-3.1-8b")
    parser.add_argument("--embeddings-dir", type=str, default="/home/jovyan/vscode-workspace/sft_ready_embeddings")
    parser.add_argument("--output-file", type=str, default="output_test_set_their.json")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    
    args = parser.parse_args()
    main(args)