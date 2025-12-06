@echo off
chcp 65001 >nul

REM 切换到脚本所在目录
cd /d "%~dp0"

echo ============================================================
echo IPPO 模型评估脚本 (带SUMO GUI可视化)
echo ============================================================
echo.
echo 使用方法:
echo   python evaluate_ippo.py ^<实验目录^> --gui --extend 100 --episodes 5
echo.
echo 示例:
echo   python evaluate_ippo.py ippo_experiments/ippo_lane_change_20251201_214543 --gui --extend 100 --episodes 5
echo.
echo ============================================================

REM 如果有命令行参数，直接运行
if not "%~1"=="" (
    echo 运行: python evaluate_ippo.py %*
    python evaluate_ippo.py %*
) else (
    REM 默认运行最新实验
    echo 未指定实验目录，请输入完整命令
    echo.
    echo 可用的实验目录:
    dir /b ippo_experiments
    echo.
)

if %errorlevel% neq 0 (
    echo.
    echo [错误] 运行失败，请检查Python环境
)

pause
