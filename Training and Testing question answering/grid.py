import argparse
import os
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, Trainer
from peft import LoraConfig, get_peft_model
import json
from tqdm import tqdm
import wandb


class MultimodalDataCollator:
    def __init__(self, tokenizer, image_folder, model):
        self.tokenizer = tokenizer
        self.image_folder = image_folder
        self.image_token_id = tokenizer.convert_tokens_to_ids('<image>')
        self.embed_weights = model.get_input_embeddings().weight.cpu()

    def __call__(self, features):
        batch_input_embeds = []
        batch_labels = []
        max_len = 0

        for feat in features:
            image_file = feat['image_file'].replace('.nii.gz', '.pt')
            image_path = os.path.join(self.image_folder, image_file)
            image_features = torch.load(image_path, map_location='cpu')

            full_text = feat['text']
            input_ids = self.tokenizer(full_text, return_tensors="pt").input_ids[0]

            try:
                image_token_index = torch.where(input_ids == self.image_token_id)[0][0]
            except IndexError:
                continue

            embeds_before = torch.nn.functional.embedding(input_ids[:image_token_index], self.embed_weights)
            embeds_after = torch.nn.functional.embedding(input_ids[image_token_index + 1:], self.embed_weights)
            combined_embeds = torch.cat([embeds_before, image_features.squeeze(0), embeds_after], dim=0)

            # Labels tensor
            labels = torch.full((combined_embeds.shape[0],), -100, dtype=torch.long)

            assistant_prompt = "assistant\n"
            assistant_token_ids = self.tokenizer(assistant_prompt, add_special_tokens=False).input_ids

            assistant_start_index_in_ids = -1
            for i in range(len(input_ids) - len(assistant_token_ids) + 1):
                if input_ids[i:i+len(assistant_token_ids)].tolist() == assistant_token_ids:
                    assistant_start_index_in_ids = i + len(assistant_token_ids)
                    break

            if assistant_start_index_in_ids != -1:
                assistant_start_in_embeds = image_token_index + image_features.shape[1] + (
                    assistant_start_index_in_ids - (image_token_index + 1)
                )
                answer_ids = input_ids[assistant_start_index_in_ids:]
                labels[assistant_start_in_embeds:assistant_start_in_embeds + len(answer_ids)] = answer_ids

            batch_input_embeds.append(combined_embeds)
            batch_labels.append(labels)
            max_len = max(max_len, combined_embeds.shape[0])

        # Pad batch
        for i in range(len(batch_input_embeds)):
            embeds = batch_input_embeds[i]
            labels = batch_labels[i]
            pad_len = max_len - embeds.shape[0]
            if self.tokenizer.padding_side == 'right':
                batch_input_embeds[i] = torch.nn.functional.pad(embeds, (0, 0, 0, pad_len))
                batch_labels[i] = torch.nn.functional.pad(labels, (0, pad_len), value=-100)
            else:
                batch_input_embeds[i] = torch.nn.functional.pad(embeds, (0, 0, pad_len, 0))
                batch_labels[i] = torch.nn.functional.pad(labels, (pad_len, 0), value=-100)

        return {
            "inputs_embeds": torch.stack(batch_input_embeds),
            "labels": torch.stack(batch_labels),
            "attention_mask": torch.ones(len(batch_input_embeds), max_len)
        }


def main(args):
    args.model_base = os.path.expanduser(args.model_base)

    # --- W&B login ---
    wandb.login()

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_base, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_special_tokens({'additional_special_tokens': ['<image>']})

    # --- Load and format dataset ---
    print(f"--> Loading dataset from {args.data_path}")
    with open(args.data_path, 'r') as f:
        data = json.load(f)

    formatted_data = []
    for item in tqdm(data, desc="Formatting Data"):
        question = item['conversations'][0]['value'].replace("<image>\n", "").strip()
        answer = item['conversations'][1]['value'].strip()
        formatted_data.append({
            "text": f"user\n<image>\n{question}\nassistant\n{answer}{tokenizer.eos_token}",
            "image_file": item['image']
        })

    dataset = Dataset.from_list(formatted_data)

    # --- Split dataset 80/10/10 ---
    dataset = dataset.train_test_split(test_size=0.2, seed=42)
    test_valid = dataset['test'].train_test_split(test_size=0.5, seed=42)
    dataset = DatasetDict({
        "train": dataset['train'],
        "validation": test_valid['train'],
        "test": test_valid['test']
    })

    # --- Save test set for later use ---
    os.makedirs(args.output_dir, exist_ok=True)
    test_json_path = os.path.join(args.output_dir, "test_set.json")
    dataset["test"].to_json(test_json_path)
    print(f"--> Test set saved to {test_json_path}")

    # --- Grid search over learning rates ---
    learning_rates = [5e-5]

    for lr in learning_rates:
        print(f"\n\n==== Training with learning rate {lr} ====\n\n")

        wandb.init(
            project="multimodal-sft-gridsearch",
            name=f"lr-{lr}",
            config={
                "learning_rate": lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "grad_accumulation": args.grad_accumulation,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "model": args.model_base
            },
            reinit=True
        )

        # --- Re-initialize model + LoRA for each LR ---
        model = AutoModelForCausalLM.from_pretrained(
            args.model_base, torch_dtype=torch.bfloat16, device_map="auto"
        )
        model.resize_token_embeddings(len(tokenizer))

        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM", bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        print("--> LoRA adapter applied.")

        # --- Training arguments ---
        training_args = TrainingArguments(
            output_dir=os.path.join(args.output_dir, f"lr-{lr}"),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accumulation,
            learning_rate=lr,
            num_train_epochs=args.epochs,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            eval_strategy="epoch",
            report_to="wandb",
            remove_unused_columns=False,
            load_best_model_at_end=True,
            metric_for_best_model="loss",
        )

        data_collator = MultimodalDataCollator(tokenizer, os.path.expanduser(args.sft_embeddings_dir), model)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"],
            data_collator=data_collator,
        )

        print("--> Starting Supervised Fine-Tuning with transformers.Trainer...")
        trainer.train()

        print("--> Evaluating best model on TEST set...")
        test_results = trainer.evaluate(eval_dataset=dataset["test"])
        print(test_results)
        wandb.log({"test_loss": test_results["eval_loss"]})

        print(f"--> Training complete for lr={lr}. Saving LoRA adapter...")
        lora_config.save_pretrained(os.path.join(args.output_dir, f"lr-{lr}"))
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom SFT for LLaVA with pre-projected embeddings.")
    parser.add_argument("--model-base", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the finetune_data.json file.")
    parser.add_argument("--sft-embeddings-dir", type=str, required=True, help="Path to the directory with the 4096-dim .pt tensors.")
    parser.add_argument("--output-dir", type=str, default="/home/jovyan/sft_adapter_5_aug")
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args)
