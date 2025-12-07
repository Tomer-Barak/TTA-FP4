import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import copy
import numpy as np
from fp4_quantization import fake_quant_fp4
import time
import json

# Configuration
MODEL_NAME = "facebook/opt-125m"
LEARNING_RATE = 1e-4
MAX_STEPS = 10
ITERATIONS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CANDIDATES = [
    # Arithmetic / Series
    ("2, 4, 6, 8, 10,", " 12"),
    ("1, 1, 2, 3, 5, 8,", " 13"),
    ("10, 9, 8, 7, 6,", " 5"),
    ("The first day is Monday, the second is Tuesday, the third is", " Wednesday"),
    
    # Associations / Knowledge
    ("The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is", " Rome"),
    ("Red is to apple as yellow is to", " banana"),
    ("Hot is to cold as up is to", " down"),
    
    # Ambiguity / Context
    ("A A B B C C D D E", " E"),
    ("x1 y1 z1 x2 y2 z2 x3 y3", " z3"),
    
    # Simple Logic
    ("Statement: It is raining. Consequence: The ground is wet. Statement: It is sunny. Consequence:", " The"), 
    
    # Repetition
    ("cat dog mouse cat dog mouse cat dog", " mouse"),
    ("1 2 3 1 2 3 1 2", " 3"),
    ("A B C A B C A B", " C"),
    ("Sun Moon Star Sun Moon Star Sun Moon", " Star"),
    ("Up Down Left Right Up Down Left", " Right"),
]

# --- FP4 Classes ---

class FP4LinearFunction(torch.autograd.Function):
    """FP4 linear layer mimicking the FP4-all-the-way NVFP4 scheme.

    - Forward GEMM: Q_RtN(W) Q_RtN(a)
    - Backward GEMM (grad_input): Q_RtN(W) Q_SR(δ)
    - Update GEMM (grad_weight): Q_SR(δ) Q_SR(a)

    Weights, activations, and neural gradients are all quantized with
    NVFP4-style settings (E2M1 data, E4M3 scales, block size 16).
    """

    @staticmethod
    def forward(ctx, input, weight, bias, meta):
        # NVFP4 configuration: block size 16, E4M3 scales, RtN in forward
        block_size = 16
        scale_format = "e4m3"

        input_q = fake_quant_fp4(
            input,
            stochastic_rounding=False,
            block_size=block_size,
            scale_format=scale_format,
        )

        weight_q = fake_quant_fp4(
            weight,
            stochastic_rounding=False,
            block_size=block_size,
            scale_format=scale_format,
        )

        # Save full-precision tensors for backward (as in the reference code)
        ctx.save_for_backward(input, weight, bias)
        ctx.block_size = block_size
        ctx.scale_format = scale_format

        output = torch.nn.functional.linear(input_q, weight_q, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        block_size = ctx.block_size
        scale_format = ctx.scale_format

        grad_input = grad_weight = grad_bias = None

        # Backward GEMM: grad_input = Q_RtN(W) Q_SR(δ)
        if ctx.needs_input_grad[0]:
            grad_output_q = fake_quant_fp4(
                grad_output,
                stochastic_rounding=True,
                block_size=block_size,
                scale_format=scale_format,
            )

            weight_q = fake_quant_fp4(
                weight,
                stochastic_rounding=False,
                block_size=block_size,
                scale_format=scale_format,
            )

            grad_input = grad_output_q.matmul(weight_q)

        # Update GEMM: grad_weight = Q_SR(δ)^T Q_SR(a)
        if ctx.needs_input_grad[1]:
            grad_output_q_t = fake_quant_fp4(
                grad_output.transpose(-2, -1),
                stochastic_rounding=True,
                block_size=block_size,
                scale_format=scale_format,
            )

            input_q = fake_quant_fp4(
                input,
                stochastic_rounding=True,
                block_size=block_size,
                scale_format=scale_format,
            )

            grad_weight = grad_output_q_t.matmul(input_q)

        if bias is not None and ctx.needs_input_grad[2]:
            # Bias is kept in higher precision; we use the full-precision gradient.
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None

class FP4Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, 
                 block_size=32, scale_format="bf16", stochastic_rounding=False):
        super().__init__(in_features, out_features, bias)
        self.meta = {
            'block_size': block_size,
            'scale_format': scale_format,
            'stochastic_rounding': stochastic_rounding
        }

    def forward(self, input):
        return FP4LinearFunction.apply(input, self.weight, self.bias, self.meta)

def replace_linear_with_fp4(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            # Create new FP4 layer
            fp4_layer = FP4Linear(
                module.in_features, 
                module.out_features, 
                module.bias is not None
            )
            # Copy weights/bias
            fp4_layer.weight.data = module.weight.data.clone()
            if module.bias is not None:
                fp4_layer.bias.data = module.bias.data.clone()
            
            # Replace
            setattr(model, name, fp4_layer)
        else:
            # Recursively replace
            replace_linear_with_fp4(module)
    return model

# --- TTA Logic ---

def entropy_loss(logits):
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    return entropy.mean()

def get_pred_token(model, tokenizer, inputs, do_sample=True):
    with torch.no_grad():
        # Generate one token
        outputs = model.generate(
            **inputs, 
            max_new_tokens=1, 
            do_sample=do_sample, 
            temperature=0.7, 
            top_k=50,
            pad_token_id=tokenizer.eos_token_id
        )
        # The last token is the prediction
        pred_token_id = outputs[0, -1].item()
        return tokenizer.decode([pred_token_id])

def run_tta(model, inputs, steps=MAX_STEPS):
    model.train() # Enable dropout for TTA
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    for _ in range(steps):
        optimizer.zero_grad()
        outputs = model(**inputs)
        # Calculate entropy of the LAST token's logits (next token prediction)
        # We want to minimize entropy of the prediction for the NEXT token
        next_token_logits = outputs.logits[:, -1, :]
        loss = entropy_loss(next_token_logits)
        loss.backward()
        optimizer.step()
    
    return model

# --- Main Stats Loop ---

def run_stats():
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model_orig = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
    
    # Results structure: {prompt_idx: { 'base': [0, 1, ...], 'tta': [...], 'tta_fp4': [...] }}
    all_results = {}
    
    print(f"\nRunning TTA Stats on {len(CANDIDATES)} examples.")
    print(f"Iterations per example: {ITERATIONS}")
    print(f"Max TTA Steps: {MAX_STEPS}")
    print("-" * 60)
    
    start_time = time.time()
    
    for i, (prompt, expected_text) in enumerate(CANDIDATES):
        print(f"\nProblem {i+1}/{len(CANDIDATES)}: '{prompt}' -> '{expected_text}'")
        start_time_per_problem = time.time()
        
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        
        problem_stats = {'base': [], 'tta': [], 'tta_fp4': []}
        
        print(f"  Starting {ITERATIONS} iterations...")
        
        for iter_idx in range(ITERATIONS):
            if (iter_idx + 1) % 10 == 0:  # Changed to every 10 for less verbosity
                print(f"  Iteration {iter_idx+1}/{ITERATIONS}...")
            
            # 1. Base Model (with sampling)
            # We use the original model in eval mode but with sampling enabled in generate
            # Actually, user said "base model has a success rate", implying variability.
            # We'll use do_sample=True.
            base_model_orig.eval()
            pred_base = get_pred_token(base_model_orig, tokenizer, inputs, do_sample=True)
            is_correct_base = expected_text.strip().lower() in pred_base.strip().lower()
            problem_stats['base'].append(is_correct_base)
            
            # 2. TTA (Standard Precision)
            model_tta = copy.deepcopy(base_model_orig)
            # TTA runs in train mode (dropout enabled)
            model_tta = run_tta(model_tta, inputs)
            model_tta.eval()
            pred_tta = get_pred_token(model_tta, tokenizer, inputs, do_sample=True)
            is_correct_tta = expected_text.strip().lower() in pred_tta.strip().lower()
            problem_stats['tta'].append(is_correct_tta)
            
            # 3. TTA (FP4)
            model_fp4 = copy.deepcopy(base_model_orig)
            model_fp4 = replace_linear_with_fp4(model_fp4)
            model_fp4 = model_fp4.to(DEVICE) # Ensure new layers are on device
            model_fp4 = run_tta(model_fp4, inputs)
            model_fp4.eval()
            pred_fp4 = get_pred_token(model_fp4, tokenizer, inputs, do_sample=True)
            is_correct_fp4 = expected_text.strip().lower() in pred_fp4.strip().lower()
            problem_stats['tta_fp4'].append(is_correct_fp4)
            
            # Cleanup to save memory
            del model_tta
            del model_fp4
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Calculate stats for this problem
        acc_base = sum(problem_stats['base']) / ITERATIONS
        acc_tta = sum(problem_stats['tta']) / ITERATIONS
        acc_fp4 = sum(problem_stats['tta_fp4']) / ITERATIONS
        
        elapsed = time.time() - start_time_per_problem
        print(f"  -> Accuracy: Base={acc_base:.2%}, TTA={acc_tta:.2%}, TTA-FP4={acc_fp4:.2%} (Time: {elapsed:.2f}s)")
        
        all_results[i] = {
            'prompt': prompt,
            'expected': expected_text,
            'acc_base': acc_base,
            'acc_tta': acc_tta,
            'acc_fp4': acc_fp4
        }
        
        # Save intermediate results
        with open('intermediate_results.json', 'w') as f:
            json.dump(all_results, f)
        print(f"  Intermediate results saved to intermediate_results.json")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS (Total Time: {total_time:.2f}s)")
    print(f"{'ID':<3} | {'Base Acc':<10} | {'TTA Acc':<10} | {'FP4 Acc':<10} | {'Prompt (Truncated)'}")
    print(f"{'-'*60}")
    
    avg_base = 0
    avg_tta = 0
    avg_fp4 = 0
    
    for i in range(len(CANDIDATES)):
        res = all_results[i]
        print(f"{i+1:<3} | {res['acc_base']:.2%}     | {res['acc_tta']:.2%}     | {res['acc_fp4']:.2%}     | {res['prompt'][:30]}...")
        avg_base += res['acc_base']
        avg_tta += res['acc_tta']
        avg_fp4 += res['acc_fp4']
        
    avg_base /= len(CANDIDATES)
    avg_tta /= len(CANDIDATES)
    avg_fp4 /= len(CANDIDATES)
    
    print(f"{'-'*60}")
    print(f"AVG | {avg_base:.2%}     | {avg_tta:.2%}     | {avg_fp4:.2%}     |")
    print(f"{'='*60}")
    
    # Save final results
    with open('final_results.json', 'w') as f:
        json.dump(all_results, f)
    print("Final results saved to final_results.json")

if __name__ == "__main__":
    run_stats()
