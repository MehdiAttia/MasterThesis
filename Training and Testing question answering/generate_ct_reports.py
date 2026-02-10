import argparse
import json
import os
import re
from collections import defaultdict
from typing import List, Dict, Any

try:
    # Optional: import transformers if available locally
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False


def safe_load_json(path: str) -> List[Dict[str, Any]]:
    """Try to be forgiving about trailing commas and stray commas. Returns a list of objects.
    Note: this is a pragmatic helper for noisy JSON files like the example you provided.
    """
    txt = open(path, 'r', encoding='utf-8').read()
    # Remove common accidental double commas
    txt = txt.replace(',,', ',')
    # Remove trailing commas before ] or }
    txt = re.sub(r',\s*([\]\}])', r'\1', txt)
    # If file is a sequence of objects (no surrounding list), try to wrap in []
    stripped = txt.strip()
    if not (stripped.startswith('[') and stripped.endswith(']')):
        txt = '[' + txt + ']'
    try:
        data = json.loads(txt)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON (attempted basic cleaning): {e}\nCleaned text snippet:\n{txt[:400]}")
    if not isinstance(data, list):
        raise ValueError('Expected a top-level JSON list of objects.')
    return data


def group_by_image(items: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, str]]]:
    groups = defaultdict(list)
    for it in items:
        img = it.get('image')
        q = it.get('question')
        a = it.get('answer')
        if not img or (not q and not a):
            continue
        groups[img].append({'question': q or '', 'answer': a or ''})
    return groups


DEFAULT_SYSTEM_PROMPT = """
You are an expert thoracic radiologist and structured-report generator. Your job is to take a set of question-and-answer pairs extracted from radiology reports (these Q&A pairs are derived from comparing a prior scan to a current scan) and produce a concise, clinically-appropriate CT chest report for the CURRENT scan.

Constraints and style:
- Use only information present in the Q&A pairs. Do NOT invent additional findings, measurements, or dates.
- Produce a structured report with these sections: Exam, Comparison, Technique, Findings, Impression.
- Findings should be written in clear, clinically-precise full sentences and may be bullet-listed when multiple distinct findings exist.
- Impression must be a short numbered list (1., 2., ...) of the most clinically relevant points, prioritizing new/worse findings and measurements mentioned in the answers.
- If the Q&A mention prior vs current changes, reflect the current status in Findings and summarize the change in the Impression when relevant (e.g., "Decrease of cardiomegaly compared with prior; currently normal heart size.").
- If a specific measurement is provided in the Q&A, include it verbatim in the Findings (do not round or re-measure).
- Keep the report professional, neutral, and avoid hedging language unless the QA used it ("may represent", "suggests").
- If Technique or other metadata are not provided, set Technique to "Not provided." in the report.

Be concise: Findings should typically be 4-10 short sentences or bullets; Impression ideally 1-5 numbered items.
"""


def build_user_prompt(image: str, qas: List[Dict[str, str]]) -> str:
    lines = [f"Image: {image}", "\nThe following question-answer pairs (derived from prior/current comparison) describe findings. Use them to create the CT report for the CURRENT scan.\n"]
    for i, qa in enumerate(qas, 1):
        q = qa.get('question','').strip()
        a = qa.get('answer','').strip()
        lines.append(f"Q{i}: {q}")
        lines.append(f"A{i}: {a}\n")
    lines.append("Produce the report now using the system instructions.")
    return '\n'.join(lines)


def generate_with_transformers(model_path: str, prompt: str, max_tokens: int = 512, temperature: float = 0.0):
    if not HF_AVAILABLE:
        raise RuntimeError('transformers or torch not available in this Python environment.')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16 if device=='cuda' else None, device_map='auto' if device=='cuda' else None)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True)
    input_ids = inputs['input_ids'].to(model.device)
    max_length = input_ids.shape[1] + max_tokens
    with torch.no_grad():
        out = model.generate(input_ids, max_length=max_length, do_sample=(temperature>0.0), temperature=temperature, top_p=0.95, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    text = tokenizer.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
    return text.strip()


import requests

def generate_via_api(api_url: str, prompt: str, max_tokens: int = 512, temperature: float = 0.0):
    # Generic POST for a text-generation API expecting JSON {"prompt":..., "max_tokens":..., ...}
    payload = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
    resp = requests.post(api_url, json=payload)
    resp.raise_for_status()
    j = resp.json()
    # Try common fields
    if 'text' in j:
        return j['text']
    if isinstance(j.get('choices'), list) and 'text' in j['choices'][0]:
        return j['choices'][0]['text']
    # Fallback: return full JSON
    return json.dumps(j)


def sanitize_filename(s: str) -> str:
    return re.sub(r'[^0-9A-Za-z._-]', '_', s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help='Path to JSON file with QA objects')
    p.add_argument('--model-path', help='Local HF-compatible model path (directory or repo id). If omitted, --api-url is required')
    p.add_argument('--api-url', help='Optional text-generation HTTP endpoint (e.g., local WebUI API)')
    p.add_argument('--output-dir', '-o', default='/home/jovyan/reports', help='Output directory')
    p.add_argument('--system-prompt-file', help='Optional file that contains the system prompt text')
    p.add_argument('--max-tokens', type=int, default=512)
    p.add_argument('--temperature', type=float, default=0.0)
    args = p.parse_args()

    if not args.model_path and not args.api_url:
        raise SystemExit('Either --model-path or --api-url must be provided.')

    data = safe_load_json(args.input)
    groups = group_by_image(data)

    os.makedirs(args.output_dir, exist_ok=True)

    system_prompt = DEFAULT_SYSTEM_PROMPT
    if args.system_prompt_file:
        system_prompt = open(args.system_prompt_file, 'r', encoding='utf-8').read()

    for image, qas in groups.items():
        user_prompt = build_user_prompt(image, qas)
        full_prompt = f"<SYSTEM>\n{system_prompt.strip()}\n</SYSTEM>\n\n<USER>\n{user_prompt}\n</USER>\n\n<ASSISTANT>\n"

        print(f'Generating report for {image} (QA pairs: {len(qas)})...')
        if args.model_path:
            report_text = generate_with_transformers(args.model_path, full_prompt, max_tokens=args.max_tokens, temperature=args.temperature)
        else:
            report_text = generate_via_api(args.api_url, full_prompt, max_tokens=args.max_tokens, temperature=args.temperature)

        report_text = report_text.strip()
        base = sanitize_filename(os.path.splitext(os.path.basename(image))[0])
        md_path = os.path.join(args.output_dir, base + '.md')
        json_path = os.path.join(args.output_dir, base + '.json')

        # Save outputs
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({'image': image, 'report': report_text, 'qas': qas}, f, ensure_ascii=False, indent=2)

        print(f'Wrote {md_path} and {json_path}')

    print('All done.')


if __name__ == '__main__':
    main()
