# Knowledge Distillation Guide

## Overview

Knowledge distillation allows you to train a small "student" model (TinyLM 42M) to mimic a large "teacher" model (e.g., Gemma-3:4B).

## Process

### Step 1: Generate Teacher Logits

Generate logits from the teacher model on your SFT dataset:

```bash
# Install mlx-lm first
pip install mlx-lm

# Generate logits (WARNING: This is computationally expensive)
# Gemma-3:4B requires significant memory (~8-16 GB)
.venv/bin/python -m tinyllm.generate_teacher_logits \
  --teacher-model google/gemma-3-4b-it \
  --train-tokens data/sft/packed_v3/train.tokens.bin \
  --train-mask data/sft/packed_v3/train.mask.bin \
  --val-tokens data/sft/packed_v3/val.tokens.bin \
  --val-mask data/sft/packed_v3/val.mask.bin \
  --output-dir data/teacher_logits \
  --batch-size 4 \
  --dtype bfloat16
```

**Expected time**: Several hours for 164k sequences
**Expected disk space**: ~50-100 GB (depending on quantization)

### Step 2: Train Student with Distillation

Train the student model using the pre-computed teacher logits:

```bash
mkdir -p logs

nohup .venv/bin/python -m tinyllm.distill \
  --base-config runs/phase1_tinystories/config.json \
  --base-checkpoint runs/phase1_tinystories/checkpoints/best \
  --train-tokens data/sft/packed_v3/train.tokens.bin \
  --train-mask data/sft/packed_v3/train.mask.bin \
  --val-tokens data/sft/packed_v3/val.tokens.bin \
  --val-mask data/sft/packed_v3/val.mask.bin \
  --output-dir runs/distill_v1 \
  --steps 2000 \
  --temperature 2.0 \
  --alpha 0.5 \
  --eval-interval 100 \
  --save-interval 250 \
  > logs/distill_v1.log 2>&1 &
```

### Step 3: Evaluate

Compare the distilled student model with:
- Original pretrained model
- Standard SFT model (without distillation)

```bash
.venv/bin/python -m tinyllm.sft_eval \
  --base-config runs/phase1_tinystories/config.json \
  --base-checkpoint runs/phase1_tinystories/checkpoints/best \
  --sft-checkpoint runs/distill_v1/checkpoints/best \
  --test-tokens data/sft/packed_v3/test.tokens.bin \
  --test-mask data/sft/packed_v3/test.mask.bin
```

## Hyperparameters

### Temperature
- **Lower (1.0-2.0)**: Teacher provides softer, more informative gradients
- **Higher (3.0-5.0)**: More aggressive smoothing of teacher predictions
- **Default**: 2.0

### Alpha
- **Lower (0.1-0.3)**: More weight on hard labels (student mostly learns from ground truth)
- **Higher (0.7-0.9)**: More weight on teacher knowledge (student mostly mimics teacher)
- **Default**: 0.5 (balanced)

## Memory Requirements

### Generating Teacher Logits
- **Gemma-3:4B**: ~8-16 GB RAM (depends on quantization)
- **Disk space**: ~50-100 GB for logits (bfloat16 quantization)

### Training Student
- **Student model (42M)**: ~0.5-1 GB RAM
- **Teacher logits (loaded in chunks)**: ~1-2 GB RAM
- **Total**: ~2-3 GB RAM

## Notes

1. **Current implementation**: The distillation script currently uses standard cross-entropy as a placeholder. To enable full distillation, you need to:
   - Modify `distill.py` to load the pre-computed teacher logits
   - Implement proper batching/chunking for the logits
   - Update the loss function to use the loaded logits

2. **Alternative approach**: Instead of logit distillation, you can use **response distillation**:
   - Generate responses from the teacher model
   - Use these as additional SFT training data
   - Simpler to implement, less memory intensive

3. **Recommended**: Start with response distillation first. If results are good, try logit distillation.

## Response Distillation (Simpler Alternative)

Generate teacher responses and use them as SFT data:

### Step 1: Generate Teacher Responses

Extract prompts from your existing SFT data and generate responses with the teacher:

```bash
# Install mlx-lm first
pip install mlx-lm

# Generate responses (WARNING: This is computationally expensive)
# Gemma-3:4B requires significant memory (~8-16 GB)
.venv/bin/python -m tinyllm.generate_teacher_responses \
  --teacher-model google/gemma-3-4b-it \
  --train-tokens data/sft/packed_v3/train.tokens.bin \
  --train-mask data/sft/packed_v3/train.mask.bin \
  --tokenizer data/tokenizer.json \
  --output data/sft/downloaded/raw/teacher_responses.jsonl \
  --max-prompts 10000 \
  --max-tokens 512 \
  --temperature 0.7
```

**Expected time**: ~2-4 hours for 10k prompts
**Expected disk space**: ~10-20 MB for responses

### Step 2: Combine with Existing SFT Data

```bash
# Add teacher responses to your existing SFT data
cat data/sft/downloaded/raw/teacher_responses.jsonl >> data/sft/downloaded/raw/combined.jsonl

# Re-pack the combined data
rm -rf data/sft/packed_v4
.venv/bin/python -m tinyllm.sft_data \
  --input data/sft/downloaded/raw/combined.jsonl \
  --tokenizer data/tokenizer.json \
  --output-dir data/sft/packed_v4 \
  --context-len 512
```

### Step 3: Train Student Model Normally

Use the standard SFT training script with the augmented data:

```bash
mkdir -p logs

nohup .venv/bin/python -m tinyllm.sft_train \
  --base-config runs/phase1_tinystories/config.json \
  --base-checkpoint runs/phase1_tinystories/checkpoints/best \
  --train-tokens data/sft/packed_v4/train.tokens.bin \
  --train-mask data/sft/packed_v4/train.mask.bin \
  --val-tokens data/sft/packed_v4/val.tokens.bin \
  --val-mask data/sft/packed_v4/val.mask.bin \
  --output-dir runs/sft_v4_teacher_responses \
  --steps 3000 \
  --eval-interval 100 \
  --save-interval 500 \
  --replay-ratio 0 \
  > logs/sft_v4_teacher_responses.log 2>&1 &
```

### Step 4: Evaluate

Compare with previous runs:

```bash
.venv/bin/python -m tinyllm.sft_eval \
  --base-config runs/phase1_tinystories/config.json \
  --base-checkpoint runs/phase1_tinystories/checkpoints/best \
  --sft-checkpoint runs/sft_v4_teacher_responses/checkpoints/best \
  --test-tokens data/sft/packed_v4/test.tokens.bin \
  --test-mask data/sft/packed_v4/test.mask.bin
```

This is much simpler and often yields similar results to logit distillation.

## Choosing a Teacher Model

### Current Context
- **Student model**: TinyLLM 42M parameters
- **Dataset**: v5 with 314,745 conversations (193M assistant tokens)
- **Hardware**: 16GB RAM, Apple Silicon
- **Goal**: Generate 10,000 teacher responses for distillation

### Teacher Model Comparison

| Model | Parameters | RAM Usage | Speed (10k prompts) | French Support | Coding | Maths | Quality |
|-------|-----------|-----------|---------------------|----------------|--------|-------|----------|
| Gemma-3-4B-it | 4B | 8-16 GB | 25-50 hours | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Excellent |
| Gemma-2-2B-it | 2B | 4-8 GB | 10-20 hours | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Very Good |
| Qwen2.5-1.5B-Instruct | 1.5B | 3-6 GB | 5-10 hours | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Very Good |
| Qwen2.5-0.5B-Instruct | 0.5B | 1-2 GB | 2-4 hours | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Good |

### Recommended Models for 16GB RAM

#### Option 1: Qwen2.5-1.5B-Instruct (Recommended)
**Best compromise between speed and quality**

```bash
# Test with 100 prompts first
.venv/bin/python -m tinyllm.generate_teacher_responses \
  --teacher-model Qwen/Qwen2.5-1.5B-Instruct \
  --train-tokens data/sft/packed_v5/train.tokens.bin \
  --train-mask data/sft/packed_v5/train.mask.bin \
  --tokenizer data/tokenizer.json \
  --output data/sft/downloaded/raw/teacher_responses_test.jsonl \
  --max-prompts 100 \
  --max-tokens 512 \
  --temperature 0.7

# If test succeeds, generate 10,000 responses
.venv/bin/python -m tinyllm.generate_teacher_responses \
  --teacher-model Qwen/Qwen2.5-1.5B-Instruct \
  --train-tokens data/sft/packed_v5/train.tokens.bin \
  --train-mask data/sft/packed_v5/train.mask.bin \
  --tokenizer data/tokenizer.json \
  --output data/sft/downloaded/raw/teacher_responses.jsonl \
  --max-prompts 10000 \
  --max-tokens 512 \
  --temperature 0.7
```

#### Option 2: Qwen2.5-0.5B-Instruct (Fastest)
**If you want maximum speed, accept slightly lower quality**

```bash
.venv/bin/python -m tinyllm.generate_teacher_responses \
  --teacher-model Qwen/Qwen2.5-0.5B-Instruct \
  --train-tokens data/sft/packed_v5/train.tokens.bin \
  --train-mask data/sft/packed_v5/train.mask.bin \
  --tokenizer data/tokenizer.json \
  --output data/sft/downloaded/raw/teacher_responses.jsonl \
  --max-prompts 10000 \
  --max-tokens 512 \
  --temperature 0.7
```

#### Option 3: Gemma-2-2B-it (High Quality)
**If you want maximum quality, accept longer generation time**

```bash
.venv/bin/python -m tinyllm.generate_teacher_responses \
  --teacher-model google/gemma-2-2b-it \
  --train-tokens data/sft/packed_v5/train.tokens.bin \
  --train-mask data/sft/packed_v5/train.mask.bin \
  --tokenizer data/tokenizer.json \
  --output data/sft/downloaded/raw/teacher_responses.jsonl \
  --max-prompts 10000 \
  --max-tokens 512 \
  --temperature 0.7
```

## Current Workflow (v5 → v6 Distillation)

### What We're Doing

We are using **response-based distillation** to improve TinyLLM's behavior:

1. **Extract prompts** from the expanded v5 dataset (314k conversations)
2. **Generate responses** using a teacher model (Gemma or Qwen)
3. **Combine** teacher responses with existing SFT data
4. **Re-pack** into v6 dataset
5. **Train** TinyLLM (student) on the augmented data
6. **Result**: TinyLLM learns to mimic the teacher's high-quality responses

### Why This Works

- Teacher models (Gemma/Qwen) are trained on massive, diverse datasets
- They know how to answer in French, code, solve math problems, etc.
- By training TinyLLM on their responses, TinyLLM learns these capabilities
- TinyLLM stays small (42M) and fast, but becomes "smarter"

### Current Status

- ✅ v5 dataset packed: 314,745 conversations, 193M tokens
- ✅ Gemma-3-4B tested but too slow (25-50 hours for 10k prompts)
- 🔄 Testing alternative teachers (Qwen2.5-1.5B recommended)
- ⏳ Next: Generate 10k teacher responses → Combine → Re-pack → Train v6

### Complete v6 Workflow

After teacher generation:

```bash
# 1. Combine teacher responses with existing data
cat data/sft/downloaded/raw/teacher_responses.jsonl >> data/sft/downloaded/raw/combined_with_teacher.jsonl

# 2. Re-pack into v6
rm -rf data/sft/packed_v6
.venv/bin/python -m tinyllm.sft_data \
  --input data/sft/downloaded/raw/combined_with_teacher.jsonl \
  --tokenizer data/tokenizer.json \
  --output-dir data/sft/packed_v6 \
  --context-len 512

# 3. Train student model (TinyLLM)
mkdir -p logs
nohup .venv/bin/python -m tinyllm.sft_train \
  --base-config runs/phase1_tinystories/config.json \
  --base-checkpoint runs/phase1_tinystories/checkpoints/best \
  --train-tokens data/sft/packed_v6/train.tokens.bin \
  --train-mask data/sft/packed_v6/train.mask.bin \
  --val-tokens data/sft/packed_v6/val.tokens.bin \
  --val-mask data/sft/packed_v6/val.mask.bin \
  --output-dir runs/sft_v6_distilled \
  --steps 3000 \
  --eval-interval 100 \
  --save-interval 500 \
  --replay-ratio 0 \
  > logs/sft_v6_distilled.log 2>&1 &

# 4. Evaluate
.venv/bin/python -m tinyllm.sft_eval \
  --base-config runs/phase1_tinystories/config.json \
  --base-checkpoint runs/phase1_tinystories/checkpoints/best \
  --sft-checkpoint runs/sft_v6_distilled/checkpoints/best \
  --test-tokens data/sft/packed_v6/test.tokens.bin \
  --test-mask data/sft/packed_v6/test.mask.bin

# 5. Test interactively
.venv/bin/python -m tinyllm.chat \
  --run-dir runs/sft_v6_distilled \
  --checkpoint best
```

### Training Time Estimates

- **Teacher generation**: 2-10 hours (depending on model choice)
- **Re-packing**: ~10-20 minutes
- **Student training**: ~2-3 hours (similar to v3, regardless of data size)
- **Total**: ~5-15 hours

### Alternative: Train v5 Directly (No Distillation)

If teacher generation takes too long, you can train directly on v5:

```bash
mkdir -p logs
nohup .venv/bin/python -m tinyllm.sft_train \
  --base-config runs/phase1_tinystories/config.json \
  --base-checkpoint runs/phase1_tinystories/checkpoints/best \
  --train-tokens data/sft/packed_v5/train.tokens.bin \
  --train-mask data/sft/packed_v5/train.mask.bin \
  --val-tokens data/sft/packed_v5/val.tokens.bin \
  --val-mask data/sft/packed_v5/val.mask.bin \
  --output-dir runs/sft_v5_large \
  --steps 5000 \
  --eval-interval 100 \
  --save-interval 500 \
  --replay-ratio 0 \
  > logs/sft_v5_large.log 2>&1 &
```

This uses 3.3x more data than v3 (193M vs 58M tokens) without distillation.
