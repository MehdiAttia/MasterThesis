import argparse
import torch
import json
import os
import sys

# --- ROBUST PATHING FIX ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
script_dir = os.path.dirname(os.path.abspath(__file__))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

from transformers import TextStreamer
import numpy as np
import tqdm

def main(args):
    # Model
    disable_torch_init()

    args.model_path = os.path.expanduser(args.model_path)
    args.model_base = os.path.expanduser(args.model_base)

    model_name = get_model_name_from_path(args.model_path)
    
    # --- SIMPLIFIED LOADING ---
    # We no longer need the image_processor, because we are not processing raw images.
    # The load_pretrained_model function will now correctly load our LlavaLlamaForCausalLM
    # which contains the mm_projector we need.
    tokenizer, model, _, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, args.load_8bit, 
        args.load_4bit, device=args.device
    )

    json_file_path = os.path.join(script_dir, 'merged_output_valid.json')
    with open(json_file_path, 'r') as file:
        data_val = json.load(file)
    output_save = []
    
    input_embedding_layer = model.get_input_embeddings()

    for element in tqdm.tqdm(data_val):
        
        args.conv_mode = "llama3"
        conv = conv_templates[args.conv_mode].copy()
        roles = conv.roles

        image_file = element["image"].replace("nii.gz", "npz")
        
        #embeddings_dir = os.path.join(script_dir, "embeddings")
        embeddings_dir = os.path.join(os.path.expanduser('~'), "train_embeddings")
        image_path = os.path.join(embeddings_dir, image_file)
        print(image_path)
        
        # This is now the PRE-PROJECTOR features from your CTViT encoder
        image_features_512d = torch.from_numpy(np.load(image_path)["arr"]).to(model.device, dtype=torch.float16)
        
        # Reshape the 5D features into a 3D tensor [batch, sequence, dim]
        num_image_tokens = image_features_512d.shape[1] * image_features_512d.shape[2] * image_features_512d.shape[3]
        image_features_512d = image_features_512d.view(image_features_512d.shape[0], num_image_tokens, image_features_512d.shape[4])

        # --- THIS IS THE CRITICAL STEP ---
        # Project the 512-dim features into the 4096-dim LLM space using the model's projector
        image_features_4096d = model.model.mm_projector(image_features_512d)

        conversations_save = []
        for conversation in element["conversations"]:
            if conversation["from"] == "human":
                inp = conversation["value"]
                if DEFAULT_IMAGE_TOKEN not in inp:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                
                conv.append_message(roles[0], inp)
                conv.append_message(roles[1], None)

                prompt = conv.get_prompt()
                
                input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

                image_token_indices = torch.where(input_ids == IMAGE_TOKEN_INDEX)[1]
                
                text_ids_before_image = input_ids[:, :image_token_indices[0]]
                text_ids_after_image = input_ids[:, image_token_indices[0] + 1:]

                embeds_before_image = input_embedding_layer(text_ids_before_image)
                embeds_after_image = input_embedding_layer(text_ids_after_image)
                
                # Combine with the NEWLY PROJECTED 4096-dim features
                final_input_embeds = torch.cat(
                    [embeds_before_image, image_features_4096d, embeds_after_image],
                    dim=1
                )

                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

                with torch.inference_mode():
                    output_ids = model.generate(
                        inputs_embeds=final_input_embeds,
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        max_new_tokens=args.max_new_tokens,
                        streamer=streamer,
                        use_cache=True
                    )

                outputs = tokenizer.decode(output_ids[0]).strip()
                conv.messages[-1][-1] = outputs

                conversations_save.append({"question": inp, "answer": outputs})

            if args.debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

        output_save.append({"image": image_file, "conversations_out": conversations_save})
    
    output_file_path = os.path.join(script_dir, "output_TRAINING_SET_inference.json")
    with open(output_file_path, "w") as json_file:
        json.dump(output_save, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="~/thesis2-workspace/CT-CLIP/models/llama_3.1_8b/models/CT-CHAT/llava-lora-llama-3.1-8b")
    parser.add_argument("--model-base", type=str, default="~/thesis2-workspace/CT-CLIP/models/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)