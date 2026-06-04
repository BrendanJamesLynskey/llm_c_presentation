# LLM Training in Raw C/CUDA

**An interactive, slide-by-slide walkthrough of Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c) — training GPT-2 / GPT-3 in simple, dependency-free C/CUDA, with no PyTorch and no cPython.**

The CPU reference, `train_gpt2.c`, is ~1,000 readable lines; the CUDA version reaches and exceeds PyTorch performance. Every stage — the parameter layout, the forward kernels, the hand-written backward pass, AdamW, and the CUDA/multi-GPU scaling — is explained with real code from the project.

---

## [Launch Presentation](https://brendanjameslynskey.github.io/llm_c_presentation/)

---

## What's Covered

| Part | Topic |
|------|-------|
| 1 | **Motivation** — why train in raw C/CUDA: no heavyweight dependencies, ~1,000 readable lines, full control, a clean reference for what PyTorch does under the hood, and the goal of reproducing GPT-2 |
| 2 | **Layout & Parameter Tensors** — the `GPT2Config`, the 16 `ParameterTensors`, one big malloc with pointer offsets, and loading weights exported from a PyTorch `.bin` checkpoint |
| 3 | **The Forward Pass Kernels** — `encoder_forward`, `layernorm_forward`, `matmul_forward`, `attention_forward`, `gelu_forward`, `residual_forward`, and the final logits + softmax + cross-entropy |
| 4 | **Backprop by Hand** — every forward kernel's matching `*_backward`, gradients that accumulate by hand, and the single grads buffer that mirrors the params |
| 5 | **The Training Step** — `gpt2_forward` → `gpt2_zero_grad` → `gpt2_backward` → `gpt2_update` (AdamW with m/v buffers), the C `main()` loop, and the dataloader over `.bin` token shards |
| 6 | **CUDA & Scaling** — porting kernels to CUDA, mixed precision (BF16), fused cuBLASLt matmuls, flash-attention, multi-GPU via NCCL and multi-node via MPI, and the performance-vs-PyTorch story |

The presentation closes with a one-slide summary grid, a note on where llm.c fits in the Zero-to-Hero arc, and key takeaways.

## Format

Built with [Reveal.js](https://revealjs.com/). Use `→` to advance, `↓` for sub-sections, and `Esc` for the slide overview. Two interactive demos: a **kernel-sequence stepper** that walks the forward → backward → update chain, and a **GPT-2 memory-budget calculator** (124M / 350M / 774M / 1558M) showing params, AdamW optimiser state, and activation estimates.

## Part of

This deck is part of [Karpathy: Neural Networks Zero to Hero](https://github.com/BrendanJamesLynskey/LLM_Hub_Karpathy_Zero_to_Hero), itself part of the [LLMs](https://github.com/BrendanJamesLynskey/LLMs) hub.

Credit: all code shown is from Andrej Karpathy's [llm.c](https://github.com/karpathy/llm.c).
