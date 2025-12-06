@echo off
chcp 65001 >nul

REM 切换到脚本所在目录
cd /d "%~dp0"

echo ============================================================
echo IPPO 模型评估脚本
echo ============================================================
echo 当前目录: %cd%

REM 设置实验目录 - 可以修改为你想测试的实验
set EXPERIMENT_DIR=ippo_experiments\ippo_lane_change_20251201_014649

REM 设置评估参数
set MODEL=final_model.pt
set EPISODES=100

echo.
echo 实验目录: %EXPERIMENT_DIR%
echo 模型: %MODEL%
echo 评估次数: %EPISODES% episodes
echo.

REM 运行评估 (不带GUI)
python evaluate_ippo.py --experiment_dir %EXPERIMENT_DIR% --model %MODEL% --episodes %EPISODES%

if %errorlevel% neq 0 (
    echo.
    echo [错误] 运行失败，请检查Python环境是否正确激活
    echo 尝试手动运行: conda activate your_env_name
)

echo.
echo ============================================================
echo 评估完成！结果保存在: %EXPERIMENT_DIR%\evaluation\
echo ============================================================
pause
