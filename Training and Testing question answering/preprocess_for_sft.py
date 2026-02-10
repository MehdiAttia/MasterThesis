import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
import sys

# --- ROBUST PATHING FIX ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from llava.model.builder import load_pretrained_model

def main(args):
    """
    Pre-processes all embeddings by passing them through the model's frozen
    multimodal projector, creating final, ready-to-use tensors for training.
    """
    print("--- Starting SFT Pre-processing ---")

    # --- Step 1: Load the full LLaVA model ---
    args.model_path = os.path.expanduser(args.model_path)
    args.model_base = os.path.expanduser(args.model_base)
    model_name = os.path.basename(args.model_path)
    
    print(f"--> Loading model from {args.model_path} to access the projector...")
    _, model, _, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name, device=args.device
    )
    projector = model.mm_projector.to(args.device)
    projector.eval()
    
    # --- START: THE DEFINITIVE FIX ---
    # 1. Automatically detect the projector's expected data type (e.g., bfloat16)
    projector_dtype = next(projector.parameters()).dtype
    print(f"--> Projector loaded. Expected dtype: {projector_dtype}")
    # --- END: THE DEFINITIVE FIX ---

    # --- Step 2: Find all source embeddings and prepare destination ---
    source_dir = os.path.expanduser(args.source_embeddings_dir)
    dest_dir = os.path.expanduser(args.dest_embeddings_dir)
    os.makedirs(dest_dir, exist_ok=True)
    
    embedding_files = [f for f in os.listdir(source_dir) if f.endswith('.npz')]
    print(f"--> Found {len(embedding_files)} source embeddings in {source_dir}.")
    print(f"--> Processed embeddings will be saved to {dest_dir}.")

    # --- Step 3: Loop, process, and save ---
    with torch.no_grad():
        for filename in tqdm(embedding_files, desc="Processing Embeddings"):
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(dest_dir, filename.replace('.npz', '.pt'))

            try:
                # --- START: THE DEFINITIVE FIX ---
                # 2. Load the numpy array first
                numpy_array = np.load(source_path)["arr"]
                
                # 3. Convert the numpy array to a Float32 tensor
                features_512d_float32 = torch.from_numpy(numpy_array).to(args.device)
                
                # 4. Now, safely cast the Float32 tensor to the projector's expected dtype
                features_512d = features_512d_float32.to(dtype=projector_dtype)
                # --- END: THE DEFINITIVE FIX ---

                # Reshape from 5D to 3D
                num_tokens = features_512d.shape[1] * features_512d.shape[2] * features_512d.shape[3]
                features_512d = features_512d.view(features_512d.shape[0], num_tokens, features_512d.shape[4])

                # Project to 4096-dim
                features_4096d = projector(features_512d)
                
                # Save the final tensor
                torch.save(features_4096d.cpu(), dest_path)
            except Exception as e:
                print(f"\n[ERROR] Failed to process file {filename}. Skipping.")
                print(f"Details: {e}")

    print("\n--- Pre-processing Complete ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process embeddings through the projector for SFT.")
    
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained LLaVA model checkpoint.")
    parser.add_argument("--model-base", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Base model from Hugging Face Hub.")
    parser.add_argument("--source-embeddings-dir", type=str, default="~/train_embeddings", help="Directory with the 512-dim .npz embeddings.")
    parser.add_argument("--dest-embeddings-dir", type=str, default="~/sft_ready_embeddings", help="Directory to save the final 4096-dim .pt tensors.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the projector on.")
    
    args = parser.parse_args()
    main(args)