#!/bin/bash
# VIDRec-MMGCN 快速启动脚本 (Linux/Mac)

echo "========================================"
echo "VIDRec-MMGCN: MMGCN for Video Recommendation"
echo "========================================"
echo ""

# 检查 Python 是否可用
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed"
    exit 1
fi

# 创建必要的目录
mkdir -p data/processed features checkpoints logs results

echo "[Step 1] Checking dependencies..."
pip install torch numpy pandas pillow tqdm scipy --quiet
echo ""

echo "[Step 2] Data Preprocessing..."
python main.py --mode preprocess
if [ $? -ne 0 ]; then
    echo "Error: Preprocessing failed"
    exit 1
fi
echo ""

echo "[Step 3] Training Model..."
python main.py --mode train
if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi
echo ""

echo "[Step 4] Evaluation..."
python main.py --mode eval
if [ $? -ne 0 ]; then
    echo "Error: Evaluation failed"
    exit 1
fi
echo ""

echo "========================================"
echo "All steps completed successfully!"
echo "========================================"
