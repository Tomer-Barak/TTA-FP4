# FP4-TTA: Test-Time Adaptation with FP4 Quantization

This repository contains code and results for evaluating Test-Time Adaptation (TTA) using Entropy Minimization on language models, with and without FP4 quantization.

## Motivation

Large Language Models deployed on edge devices are typically quantized to FP4 for memory and compute efficiency. However, quantization can degrade accuracy. This work investigates whether **Test-Time Adaptation (TTA)** can improve predictions at inference time, even when operating under FP4 quantization constraints.

We combine two recent advances:
- **TENT** (Wang et al., ICLR 2021): Test-time entropy minimization for model adaptation
- **FP4 All The Way** (Chmiel et al., 2025): Fully quantized training with 4-bit precision

## Overview

We evaluate three configurations on 15 sequence completion tasks:
- **Base Model**: OPT-125M without adaptation (with sampling)
- **TTA**: Entropy Minimization with full precision
- **TTA-FP4**: Entropy Minimization with FP4-quantized weights

**Key Results:**
| Configuration | Accuracy (95% CI) |
|--------------|-------------------|
| Base Model   | 27.5% (25.3%, 29.9%) |
| TTA          | 40.1% (37.7%, 42.6%) |
| TTA-FP4      | 32.1% (29.8%, 34.5%) |

See [REPORT.md](REPORT.md) for full experimental details and analysis.

## Repository Structure

```
├── README.md                 # This file
├── REPORT.md                 # Full experimental report with analysis
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
├── src/
│   ├── __init__.py
│   ├── fp4_quantization.py   # FP4 quantization primitives
│   └── stats_tta.py          # Main experiment script
├── results/
│   └── final_results.json    # Raw experimental results
└── references/
    └── fp4_all_the_way.pdf   # Chmiel et al. (2025) - FP4 training paper
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd fp4-tta

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Running Experiments

To reproduce the experiments:

```bash
cd src
python stats_tta.py
```

**Note:** The full experiment runs 100 iterations for each of 15 tasks across 3 conditions (4,500 total evaluations). On CPU, this takes several hours. On GPU, it completes faster.

### Configuration

You can modify the following parameters in `src/stats_tta.py`:

```python
MODEL_NAME = "facebook/opt-125m"  # HuggingFace model
LEARNING_RATE = 1e-4              # TTA learning rate
MAX_STEPS = 10                    # TTA optimization steps per sample
ITERATIONS = 100                  # Trials per task (for statistics)
```

### Output

The script produces:
- `intermediate_results.json`: Saved after each task (for crash recovery)
- `final_results.json`: Complete results after all tasks

## Method

### Test-Time Adaptation (TTA)

We use entropy minimization as the TTA objective:

$$L = -\sum_{i} p_i \log(p_i)$$

where $p_i$ is the softmax probability of the next token prediction. Minimizing entropy encourages the model to become more confident about its predictions.

### FP4 Quantization

We simulate FP4 (E2M1 format) computation using:
- Fake quantization during forward pass
- Straight-Through Estimator (STE) for gradients
- Block-wise quantization (block size: 32)
- BF16 scaling factors

## Results

See [results/final_results.json](results/final_results.json) for raw data and [REPORT.md](REPORT.md) for full analysis including:
- Per-task accuracy breakdown with 95% Wilson confidence intervals
- Statistical significance testing
- Discussion of when TTA helps vs. hurts

## Citation

If you use this code, please cite:

```bibtex
@misc{fp4tta2025,
  title={Test-Time Adaptation with FP4 Quantization},
  author={Tomer Barak}
  year={2025},
  url={https://github.com/Tomer-Barak/TTA-FP4}
}
```

## License

MIT License
