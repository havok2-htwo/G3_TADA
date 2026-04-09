import os
import time
import torch
import soundfile as sf
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()
from backend.vendor.tada.modules.tada import TadaForCausalLM, InferenceOptions
from dataclasses import dataclass
from typing import Any

@dataclass
class Prompt:
    audio: Any = None
    audio_len: Any = None
    token_positions: Any = None
    token_values: Any = None
    text_tokens: Any = None
    text_tokens_len: Any = None
    encoded_expanded: Any = None
    non_sampled_encoded_expanded: Any = None
    token_masks: Any = None
    text: Any = None
    sample_rate: Any = 24000
    
# Config
ROOT = r"x:\dev\G3_TADA3B"
# Model name (can be changed to 'HumeAI/tada-1b' for the smaller model)
MODEL_NAME = "HumeAI/tada-1b"
DEVICE = "cuda:0"
# Define batch sizes to benchmark
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
OUT_DIR = Path(ROOT) / "benchmark_results"

# 16 sentences for scaling up
SENTENCES = [
    "The weather is really excellent today, don't you think?",
    "Artificial intelligence is developing rapidly and opening up new possibilities.",
    "I am a virtual voice assistant and I am happy to help you with your tasks.",
    "Batch processing on graphics cards saves a massive amount of time in production.",
    "Let us work together to find out how we can solve this problem efficiently.",
    "Many technology companies are increasingly relying on locally running open-source models.",
    "A good language model must not only be intelligent, but also sound natural.",
    "I look forward to presenting this fantastic audio result to you.",
    "Have you ever thought about how many calculations take place here per second?",
    "Sometimes it is better to take a deep breath before making a decision.",
    "Patience is a virtue, especially when training complex neural networks.",
    "Do not forget to thoroughly test the project before calling it a day.",
    "When we work together, we can overcome even the most difficult challenges.",
    "I can hardly wait for us to release the final version of the software.",
    "Speech synthesis on an RTX 5090 is a true demonstration of technical brilliance.",
    "Enjoy this audio sample and the quality of the generated waveform."
]


def load_model():
    hf_token = os.getenv("HF_TOKEN")
    print(f"Loading {MODEL_NAME}...")
    model = TadaForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, low_cpu_mem_usage=True, token=hf_token,
    )
    model = model.to(DEVICE)
    model.decoder.to(DEVICE)
    model.eval()
    return model

def main():
    model = load_model()
    
    # Resolve the first voice prompt cache available
    voice_dir = Path(ROOT) / "backend" / "data" / "voices"
    voice_path = next(d for d in voice_dir.iterdir() if (d / "prompt_cache.pt").exists())
    cache_file = voice_path / "prompt_cache.pt"
    
    print(f"Loaded prompt cache from: {cache_file.name}")
    p_dict = torch.load(cache_file, map_location=DEVICE)
    
    prompt = Prompt()
    for k, v in p_dict.items():
        if hasattr(prompt, k) and isinstance(v, torch.Tensor):
            setattr(prompt, k, v)

    
    # Let's verify sample rate
    sample_rate = getattr(model.decoder, "sample_rate", None)
    if isinstance(sample_rate, int):
        sr = sample_rate
    else:
        sr = getattr(model.decoder.config, "sample_rate", 24000)

    print(f"Resolved Sample Rate: {sr}")
    opts = InferenceOptions(
        text_repetition_penalty=1.0,
        num_flow_matching_steps=20
    )

    for b in BATCH_SIZES:
        batch_out_dir = OUT_DIR / str(b)
        batch_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Replicate sentences if b > len(SENTENCES)
        texts = []
        while len(texts) < b:
            texts.extend(SENTENCES)
        texts = texts[:b]
        
        # Build batched prompt correctly as a proper object with batched token_values
        prompt_batched = Prompt()
        if hasattr(prompt, "token_values") and prompt.token_values is not None:
             # token_values originally [1, L, 512]. expand to [b, L, 512]
            prompt_batched.token_values = prompt.token_values.expand(b, -1, -1).clone().to(DEVICE)
            
        # Copy other tensors needed for prompt if they exist
        for field in prompt.__dataclass_fields__:
            val = getattr(prompt, field)
            if isinstance(val, torch.Tensor) and field != "token_values":
                if val.shape[0] == 1:
                    setattr(prompt_batched, field, val.expand(b, *val.shape[1:]).clone().to(DEVICE))
                else:
                    setattr(prompt_batched, field, val.to(DEVICE))
        # Read text from metadata.json of the voice
        import json
        with open(voice_path / "metadata.json", "r", encoding="utf-8") as f:
            voice_meta = json.load(f)
            prompt_batched.text = [voice_meta["transcript"]] * b
            prompt.text = [voice_meta["transcript"]]
            
        print(f"\n=========================================")
        print(f"Running Benchmark for BATCH_SIZE = {b}")
        print(f"=========================================")
        
        start_time = time.time()
        
        try:
            with torch.inference_mode():
                out = None
                torch.cuda.reset_peak_memory_stats(DEVICE)
                try:
                    out = model.generate(prompt=prompt_batched, text=texts, inference_options=opts)
                except Exception as eval_e:
                    import traceback
                    print(f"CRASH INSIDE MODEL.GENERATE FOR B={b}:")
                    print(traceback.format_exc())
                    continue
                    
                audio_outputs = []
                if hasattr(out, "audio") and out.audio:
                    audio_outputs.extend(out.audio)
            
            elapsed = time.time() - start_time
            vram_peak_mb = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)
            print(f"Batch {b} completed in {elapsed:.2f} seconds! Peak VRAM: {vram_peak_mb:.1f} MB")
            
            for idx, (sentence, audio) in enumerate(zip(texts, audio_outputs)):
                # audio shape is [C, T] or [1, T]
                if audio.ndim == 2 and audio.shape[0] == 1:
                    audio_np = audio.squeeze(0).float().cpu().numpy()
                elif audio.ndim == 1:
                    audio_np = audio.float().cpu().numpy()
                else:
                    audio_np = audio.T.float().cpu().numpy() # [T, C] for channels
                    
                file_path = batch_out_dir / f"item_{idx}.wav"
                sf.write(str(file_path), audio_np, sr)
                print(f"   -> Saved {file_path.name}")
                
        except Exception as e:
            print(f"Batch {b} failed with error:")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
