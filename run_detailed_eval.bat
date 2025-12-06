@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ========================================
echo 详细评估 - 记录直行车辆轨迹
echo ========================================

REM 设置实验目录（修改为你的最新实验目录）
set EXPERIMENT_DIR=ippo_experiments\ippo_lane_change_20251201_171553

REM 运行详细评估
python evaluate_detailed.py --experiment_dir %EXPERIMENT_DIR% --model final_model.pt

echo.
echo Done! Results saved in experiment directory.
pause
