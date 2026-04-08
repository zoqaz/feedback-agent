# feedback-agent
feedback agent to parse routines and provide feedback

Recommended model:
- Qwen2.5-7B-Instruct (Q4_K_M)
- Source: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF

Download:
```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
  qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf \
  qwen2.5-7b-instruct-q4_k_m-00002-of-00002.gguf \
  --local-dir models/qwen2.5-7b-gguf