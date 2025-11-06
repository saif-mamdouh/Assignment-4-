#!/usr/bin/env bash
set -e


# Ensure HF login is done as team leader (token available)
# 1) Download checkpoints and run inference
python assignment_4/run_inference.py --hf-repo your-hf-username/seqtrack-assignment-3 --seqs-file assignment_3/selected_test_sequences.txt


# 2) After completion, inspect outputs_inference/metrics_by_checkpoint.json
python -c "import json; print(json.load(open('assignment_4/outputs_inference/metrics_by_checkpoint.json'), indent=2))"