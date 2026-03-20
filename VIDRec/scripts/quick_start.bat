@echo off
REM VIDRec-MMGCN 快速启动脚本 (Windows)

echo ========================================
echo VIDRec-MMGCN: MMGCN for Video Recommendation
echo ========================================
echo.

REM 检查 Python 是否可用
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM 创建必要的目录
if not exist "data\processed" mkdir "data\processed"
if not exist "features" mkdir "features"
if not exist "checkpoints" mkdir "checkpoints"
if not exist "logs" mkdir "logs"
if not exist "results" mkdir "results"

echo [Step 1] Checking dependencies...
pip install torch numpy pandas pillow tqdm scipy --quiet
echo.

echo [Step 2] Data Preprocessing...
python main.py --mode preprocess
if errorlevel 1 (
    echo Error: Preprocessing failed
    pause
    exit /b 1
)
echo.

echo [Step 3] Training Model...
python main.py --mode train
if errorlevel 1 (
    echo Error: Training failed
    pause
    exit /b 1
)
echo.

echo [Step 4] Evaluation...
python main.py --mode eval
if errorlevel 1 (
    echo Error: Evaluation failed
    pause
    exit /b 1
)
echo.

echo ========================================
echo All steps completed successfully!
echo ========================================
pause
