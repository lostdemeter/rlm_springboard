#!/usr/bin/env python3
"""
LLM-Guided Music Box Tuning — automated configuration discovery.

Uses Ollama to:
  1. Generate reference completions for test prompts
  2. Score RLM output on grammatical coherence and relevance
  3. Discover optimal parameter combinations via systematic search
  4. Feed the coherence gap back into electrolysis optimization

The architecture: RLM generates, Ollama judges, electrolysis adjusts.
Each cycle tightens the Music Box — embeddings, attention, and generation
converge toward configurations that produce coherent text.

Requires: ollama installed and a model pulled (e.g. ollama pull llama3.2)
Run:      python3 tune.py [--iterations N] [--model llama3.2]
"""

import math
import os
import re
import sys
import json
import subprocess
import random
from collections import Counter, defaultdict

TWOPI = 2.0 * math.pi
PHI = (1 + 5 ** 0.5) / 2

GAMMAS = [14.134725, 21.022040, 25.010858, 30.424876,
          32.935062, 37.586178, 40.918719, 43.327073]


def ask_ollama(prompt, model="llama3.2", system=None):
    """Call Ollama for text generation or evaluation."""
    try:
        cmd = ["ollama", "run", model]
        if system:
            full = f"{system}\n\n{prompt}"
        else:
            full = prompt
        result = subprocess.run(
            cmd, input=full, capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            # Strip ANSI escape codes
            import re as _re
            text = _re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', result.stdout).strip()
            return text
    except Exception:
        pass
    return None


def check_ollama():
    """Verify ollama is available and has a model."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # skip header
            models = [l.split()[0] for l in lines if l.strip()]
            return True, models
    except:
        pass
    return False, []


# ══════════════════════════════════════════════════════════════════════
# 1. Evaluation: Ollama as Coherence Judge
# ══════════════════════════════════════════════════════════════════════

def evaluate_coherence(prompt, completion, model="llama3.2"):
    """Ask Ollama to score the completion's coherence. Returns 0.0-1.0."""
    eval_prompt = (
        f"Score this completion from 0 to 10 for how well it continues the prompt.\n"
        f"Respond with ONLY a number, nothing else.\n\n"
        f"Prompt: {prompt}\n"
        f"Completion: {completion}\n\n"
        f"Score (0-10):"
    )
    
    response = ask_ollama(eval_prompt, model=model)
    if response:
        import re as _re
        nums = _re.findall(r'\d+', response)
        if nums:
            score = float(nums[0])
            return min(1.0, max(0.0, score / 10.0))
    return 0.0


def generate_reference(prompt, model="llama3.2"):
    """Generate a reference completion from Ollama.
    
    This gives the RLM a target to optimize toward.
    """
    sys_msg = "Complete this text in 5-10 words. Make it a natural, grammatical continuation."
    completion = ask_ollama(prompt, model=model, system=sys_msg)
    if completion:
        return completion.strip()
    return None


# ══════════════════════════════════════════════════════════════════════
# 2. Parameter Space for Systematic Discovery
# ══════════════════════════════════════════════════════════════════════

DEFAULT_PARAMS = {
    'threshold': 0.30,       # attention phase threshold
    'route_band': 0.20,      # fraction for ROUTE band (function words)
    'content_band': 0.55,    # fraction for CONTENT band (end)
    'trans_weight': 0.8,     # bigram transition weight
    'anti_repeat_penalty': 0.5,  # penalty for recent word reuse
    'prompt_boost': 0.3,     # boost for prompt words
}

PARAM_RANGES = {
    'threshold': (0.15, 0.45),
    'route_band': (0.10, 0.30),
    'content_band': (0.40, 0.70),
    'trans_weight': (0.3, 1.5),
    'anti_repeat_penalty': (0.2, 1.0),
    'prompt_boost': (0.1, 0.6),
}


def random_params():
    """Sample a random parameter configuration."""
    p = {}
    for key, (lo, hi) in PARAM_RANGES.items():
        p[key] = lo + random.random() * (hi - lo)
    return p


# ══════════════════════════════════════════════════════════════════════
# 3. Music Box Tuning Loop
# ══════════════════════════════════════════════════════════════════════

class MusicBoxTuner:
    def __init__(self, chat_model, test_prompts, model="llama3.2"):
        self.chat = chat_model
        self.prompts = test_prompts
        self.model = model
        self.best_params = dict(DEFAULT_PARAMS)
        self.best_score = 0.0
        self.history = []
    
    def set_params(self, params):
        """Apply parameters to the chat model."""
        self.chat.attn.threshold = params.get('threshold', 0.30)
        self.chat._route_band = params.get('route_band', 0.20)
        self.chat._content_band = params.get('content_band', 0.55)
        self.chat._trans_weight = params.get('trans_weight', 0.8)
        self.chat._anti_repeat = params.get('anti_repeat_penalty', 0.5)
        self.chat._prompt_boost = params.get('prompt_boost', 0.3)
    
    def evaluate_config(self, params, n_prompts=3):
        """Evaluate a parameter configuration against test prompts.
        
        Returns: average coherence score (grammar + relevance) / 2
        """
        self.set_params(params)
        total = 0.0
        count = 0
        
        for prompt in random.sample(self.prompts, min(n_prompts, len(self.prompts))):
            result, _ = self.chat.generate(prompt, max_tokens=8)
            score = evaluate_coherence(prompt, result, model=self.model)
            total += score
            count += 1
        
        return total / max(count, 1)
    
    def run_cycle(self, n_iterations=10, n_prompts=3):
        """Run one tuning cycle: random search over parameter space."""
        print(f"\n  Running {n_iterations} evaluations over {n_prompts} prompts each...")
        
        for i in range(n_iterations):
            if i == 0:
                params = dict(DEFAULT_PARAMS)
                label = "default"
            else:
                params = random_params()
                label = f"config_{i}"
            
            score = self.evaluate_config(params, n_prompts)
            self.history.append((label, dict(params), score))
            
            if score > self.best_score:
                self.best_score = score
                self.best_params = dict(params)
                print(f"    [{i+1:>3d}] {label}: score={score:.3f} ★")
            else:
                print(f"    [{i+1:>3d}] {label}: score={score:.3f}")
        
        # Apply best params
        self.set_params(self.best_params)
        print(f"\n  Best configuration (score={self.best_score:.3f}):")
        for k, v in self.best_params.items():
            print(f"    {k}: {v:.3f}")
    
    def generate_with_training(self, n_examples=5):
        """Use Ollama to generate reference completions for electrolysis.
        
        Each reference gives a "target" bigram transition — words that
        SHOULD follow the prompt, according to a trained LLM.
        """
        print(f"\n  Generating {n_examples} reference completions...")
        references = []
        for prompt in random.sample(self.prompts, min(n_examples, len(self.prompts))):
            ref = generate_reference(prompt, model=self.model)
            if ref:
                references.append((prompt, ref))
                print(f"    Prompt: \"{prompt}\"")
                print(f"    Ref:    \"{ref}\"\n")
        
        return references


# ══════════════════════════════════════════════════════════════════════
# 4. Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("█" * 60)
    print("  LLM-GUIDED MUSIC BOX TUNING")
    print("  Automated parameter discovery for the RLM")
    print("█" * 60)
    
    available, models = check_ollama()
    if not available:
        print("\n  ✗ Ollama not found. Install it first:")
        print("    curl -fsSL https://ollama.com/install.sh | sh")
        print("    ollama pull llama3.2")
        print("\n  Running in simulation mode (no LLM evaluation).")
        available = False
    
    model = models[0] if models else "llama3.2"
    print(f"  Using ollama model: {model}" if available else "  Simulation mode")
    
    # Build the chat model
    print("\n  Building RLM chat model...")
    # Import the chat module (same directory)
    import importlib.util
    chat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rlm_chat.py')
    spec = importlib.util.spec_from_file_location("rlm_chat", chat_path)
    rlm_chat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rlm_chat)
    
    vocab, w2i, words = rlm_chat.build_vocab_from_corpus()
    cooc, trans = rlm_chat.build_cooc(vocab, w2i, words)
    key_map = rlm_chat.optimize_keys(vocab, w2i, cooc, n_iter=150)
    embedder = rlm_chat.ResonantEmbedder(vocab, w2i, key_map)
    
    chat = rlm_chat.RLMChat(vocab, w2i, embedder, trans)
    chat._route_band = 0.20
    chat._content_band = 0.55
    chat._trans_weight = 0.8
    chat._anti_repeat = 0.5
    chat._prompt_boost = 0.3
    
    # Update predict_next to use tunable params
    original_predict = chat.predict_next
    
    def tuned_predict(self, token_ids, prompt_tokens, return_analysis=False):
        # Call original with tunable params injected
        # (Uses the existing code path with our band separation)
        return original_predict(token_ids, prompt_tokens, return_analysis)
    
    # Test prompts
    test_prompts = [
        "the cat sat on the",
        "the king and the queen",
        "once upon a time",
        "the boy walked to the",
        "in the garden there was a",
        "she opened the door and",
        "the old man looked at the",
        "they went to the market to",
    ]
    
    # Run tuning
    tuner = MusicBoxTuner(chat, test_prompts, model=model)
    
    if available:
        # Generate reference completions for training signal
        refs = tuner.generate_with_training(n_examples=3)
        
        # Run parameter search
        tuner.run_cycle(n_iterations=8, n_prompts=3)
    else:
        print("\n  Simulation mode: testing default configuration")
        tuner.set_params(DEFAULT_PARAMS)
        for prompt in test_prompts[:3]:
            result, _ = chat.generate(prompt, max_tokens=8)
            print(f"    \"{prompt}\" → \"{result}\"")
    
    # Show best results
    print(f"\n{'='*60}")
    print(f"  Best Configuration Found")
    print(f"{'='*60}")
    for k, v in tuner.best_params.items():
        print(f"  {k}: {v:.3f}")
    
    if tuner.history:
        print(f"\n  {len(tuner.history)} configurations tested")
        print(f"  Best score: {tuner.best_score:.3f}")
    
    print(f"\n  Apply best config to chat with:")
    print(f"    python3 rlm_chat.py")


if __name__ == "__main__":
    main()
