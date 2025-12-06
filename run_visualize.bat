@echo off
echo ============================================================
echo IPPO Model Visualization in SUMO GUI (Fixed Version)
echo ============================================================
echo.
echo Model: final_model.pt (99%% success rate)
echo Experiment: ippo_lane_change_20251202_211111
echo.

cd /d "%~dp0"

REM Use the fixed visualization script
python visualize_best_model.py ippo_experiments/ippo_lane_change_20251202_211111 --episodes 3 --delay 0.03

pause
