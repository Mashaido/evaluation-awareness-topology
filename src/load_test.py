"""Load the Nemotron 49B base model plus the Hua adapter on one GPU slice and test it.

Purpose: find out which loading route works on a 48 GB slice, how fast it is, and
whether the hidden state extraction path works. It does not use the authors' data.

Usage (inside the container, with ~/envs/ea first on PYTHONPATH):
    python3 -u src/load_test.py --mode 4bit
    python3 -u src/load_test.py --mode bf16offload
"""
import argparse
import glob
import os
import sys
import time
import traceback

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "nvidia/Llama-3_3-Nemotron-Super-49B-v1"
ADAPTER = "timhua/wood_v2_sftr4_filt"
LAYERS = [10, 14, 18, 22, 26, 30]


def snapshot(repo):
    hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/hf_home")), "hub")
    found = sorted(glob.glob(os.path.join(hub, "models--" + repo.replace("/", "--"), "snapshots", "*")))
    if not found:
        raise FileNotFoundError("no snapshot for " + repo + " under " + hub)
    return found[-1]


def gb(n):
    return round(n / 1e9, 1)


def report_gpu(tag):
    torch.cuda.synchronize()
    print(tag, "| GPU allocated GB", gb(torch.cuda.memory_allocated()),
          "| peak GB", gb(torch.cuda.max_memory_allocated()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["4bit", "bf16offload"], required=True)
    ap.add_argument("--gpu-gib", type=float, default=44.0, help="GPU memory the offload route may use")
    ap.add_argument("--cpu-gib", type=float, default=100.0, help="CPU memory the offload route may use")
    ap.add_argument("--new-tokens", type=int, default=24)
    args = ap.parse_args()

    print("torch", torch.__version__, "| GPU", torch.cuda.get_device_name(0), flush=True)
    base_path = snapshot(BASE)
    adapter_path = snapshot(ADAPTER)
    print("base:", base_path, flush=True)
    print("adapter:", adapter_path, flush=True)

    tok = AutoTokenizer.from_pretrained(base_path)

    t0 = time.time()
    if args.mode == "4bit":
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_path, quantization_config=cfg, device_map={"": 0},
            torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, device_map="auto",
            max_memory={0: str(int(args.gpu_gib)) + "GiB", "cpu": str(int(args.cpu_gib)) + "GiB"},
            trust_remote_code=True,
        )
    print("base model loaded in", round(time.time() - t0), "s", flush=True)
    report_gpu("after base")

    from peft import PeftModel
    t1 = time.time()
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    print("adapter loaded in", round(time.time() - t1), "s", flush=True)
    report_gpu("after adapter")

    messages = [
        {"role": "system", "content": "detailed thinking off"},
        {"role": "user", "content": "Write a Python function that adds two numbers."},
    ]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("CHAT TEMPLATE (first 400 characters):", repr(text[:400]), flush=True)
    inputs = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda:0")
    n_in = inputs["input_ids"].shape[1]
    print("prompt tokens:", n_in, flush=True)

    # Hidden state extraction path: one forward pass, last prompt token, six layers.
    t2 = time.time()
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs = out.hidden_states
    print("forward pass in", round(time.time() - t2, 1), "s | hidden state tensors:", len(hs),
          "| shape of one:", tuple(hs[0].shape), flush=True)
    for layer in LAYERS:
        v = hs[layer][0, -1].float()
        print("layer", layer, "last token vector: size", v.numel(), "norm", round(v.norm().item(), 1),
              "finite", bool(torch.isfinite(v).all()), flush=True)
    report_gpu("after forward")

    t3 = time.time()
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=args.new_tokens, do_sample=False)
    dt = time.time() - t3
    print("generated", args.new_tokens, "tokens in", round(dt, 1), "s =",
          round(args.new_tokens / dt, 2), "tokens per second", flush=True)
    print("OUTPUT:", repr(tok.decode(gen[0][n_in:], skip_special_tokens=True)), flush=True)
    report_gpu("after generate")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
