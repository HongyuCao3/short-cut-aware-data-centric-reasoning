#!/bin/bash
# Run full pretrained GPT-2 experiments on all 3 new datasets sequentially
# GPU 1, nohup background

set -e
source /home/local/ASURITE/hongyuca/miniconda3/bin/activate sart
export CUDA_VISIBLE_DEVICES=1

cd /home/local/ASURITE/hongyuca/short-cut-aware-data-centric-reasoning

echo "========================================"
echo "Starting new dataset experiments"
echo "$(date)"
echo "========================================"

echo ""
echo "[1/3] Running AQuA-RAT..."
echo "$(date)"
python3 run_pretrained.py --dataset aqua

echo ""
echo "[2/3] Running SVAMP..."
echo "$(date)"
python3 run_pretrained.py --dataset svamp

echo ""
echo "[3/3] Running StrategyQA..."
echo "$(date)"
python3 run_pretrained.py --dataset strategyqa

echo ""
echo "========================================"
echo "All experiments completed!"
echo "$(date)"
echo "========================================"
