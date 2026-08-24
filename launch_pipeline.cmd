@echo off
cd /d "%~dp0"
call ".venv\Scripts\activate.bat"

echo [pipeline] stage 1: supervised pretraining on Lichess human games...
python -u pretrain_supervised.py > logs\pipeline.log 2>&1
if errorlevel 1 (
    echo [pipeline] PRETRAIN FAILED - see logs\pipeline.log
    exit /b 1
)

echo [pipeline] stage 2: self-play fine-tuning (resumes from pretrained model)...
python -u run_training.py 6 --device auto --iterations 700 >> logs\pipeline.log 2>&1
if errorlevel 1 (
    echo [pipeline] FINE-TUNE EXITED NONZERO - wrapper may have hit retry limit
    exit /b 1
)
echo [pipeline] ALL DONE
