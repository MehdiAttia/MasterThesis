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

    print(f"--> Loading TEXT-ONLY base model from: {args.model_base}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token  # align pad token
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
    print(f"Model device: {model.device}, dtype: {next(model.parameters()).dtype}")

    # --- Load validation JSON ---
    json_file_path = os.path.join(os.path.dirname(__file__), 'merged_output_valid.json')
    try:
        with open(json_file_path, 'r') as file:
            data_val = json.load(file)
    except FileNotFoundError:
        print(f"[FATAL] Missing validation file: {json_file_path}")
        return

    # DEBUG MODE: restrict sample size
    data_to_process = data_val
    print(f"\n--- DEBUG MODE: Processing only {len(data_to_process)} samples ---")

    # --- Load embeddings ---
    print("--> Pre-loading image embeddings...")
    image_embeddings = {}
    for element in tqdm(data_to_process, desc="Loading embeddings"):
        image_key = element["image"]
        image_file = image_key.replace(".nii.gz", ".pt")
        image_path = os.path.join(args.embeddings_dir, image_file)

        if os.path.exists(image_path):
            emb = torch.load(image_path, map_location=model.device)
            emb = emb.to(dtype=next(model.parameters()).dtype)
            image_embeddings[image_key] = emb
        else:
            print(f"[WARNING] Missing embedding for {image_key} at {image_path}")

    print(f"--> {len(image_embeddings)} embeddings loaded onto {model.device}")

    # --- Inference loop ---
    output_save = []
    img_token_id = tokenizer.convert_tokens_to_ids('[IMG]')
    pad_token_id = tokenizer.pad_token_id

    print("\nRunning inference with debugging...")
    for element in tqdm(data_to_process, desc="Inference"):
        if element["image"] not in image_embeddings:
            continue

        img_emb = image_embeddings[element["image"]]

        for conversation in element["conversations"]:
            if conversation["from"] != "human":
                continue

            question = conversation["value"].replace("<image>\n", "").strip()
            prompt = f"user\n[IMG]\n{question}\nassistant\n"

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

            # Find [IMG] token
            try:
                img_token_index = torch.where(input_ids == img_token_id)[1][0]
            except IndexError:
                print(f"[DEBUG] Skipping: No [IMG] token found in: {prompt}")
                continue

            input_embed_layer = model.get_input_embeddings()
            embeds_before = input_embed_layer(input_ids[:, :img_token_index])
            embeds_after = input_embed_layer(input_ids[:, img_token_index + 1:])
            final_input_embeds = torch.cat([embeds_before, img_emb, embeds_after], dim=1)

            # Attention mask
            attention_mask = torch.ones(final_input_embeds.shape[:2], dtype=torch.long, device=model.device)

            # --- Generation ---
            with torch.inference_mode():
                output_ids = model.generate(
                    inputs_embeds=final_input_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,       # GREEDY decoding
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=pad_token_id
                )

            # --- DEBUG OUTPUT ---
            print("\n--- DEBUG: Generation Output ---")
            print(f"Prompt text:\n{prompt}\n")
            print(f"Prompt length (embeds): {final_input_embeds.shape[1]}")
            print(f"Raw output_ids length: {len(output_ids[0])}")
            print(f"First 50 raw token IDs: {output_ids[0][:50].tolist()}")

            # --- FIXED RESPONSE EXTRACTION ---
            decoded_full = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            answer = decoded_full.replace(prompt, "").strip()  # remove prompt, keep generated text

            print(f"Decoded (clean): {answer[:300]!r}")
            print("--- END DEBUG ---\n")

            output_save.append({
                "image": element["image"],
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
    parser.add_argument("--model-base", type=str,
                        default="~/thesis2-workspace/CT-CLIP/models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--model-path", type=str, default="/home/jovyan/sft_adapter_5_aug/lr-5e-05/checkpoint-44150")
    parser.add_argument("--embeddings-dir", type=str, default="/home/jovyan/vscode-workspace/sft_ready_embeddings")
    parser.add_argument("--output-file", type=str, default="output_SFT_finetuned_results_augmented.json")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    main(args)
