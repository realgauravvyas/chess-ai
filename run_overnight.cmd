@echo off
REM ===================================================================
REM  v7: corrected self-play fine-tune from the supervised checkpoint.
REM
REM  Writes only into runs\v7\ and logs\train_v7.log, so every existing
REM  checkpoint and log is left untouched.
REM
REM  Differences from the v5 run that regressed:
REM    * mirror augmentation no longer corrupts castling planes
REM    * MCTS simulations 40 -> 128 (sharper, less noisy policy targets)
REM    * gradient steps 200 -> 80 (less overfitting to noisy targets)
REM    * learning rate 3e-4 -> 2e-4
REM    * AlphaGo-Zero gating: self-play always uses the best weights and
REM      new weights are promoted only after winning a head-to-head match
REM    * every gate point also logs a match vs the frozen pretrained net
REM ===================================================================
setlocal
set ROOT=%~dp0

"%ROOT%.venv\Scripts\python.exe" -u "%ROOT%run_training.py" ^
  --run-dir "%ROOT%runs\v7" ^
  --seed-from "%ROOT%checkpoints_v7_baseline\iter_200.pt" ^
  6 ^
  --device auto ^
  --iterations 400 ^
  --sims 128 ^
  --games 12 ^
  --train-steps 80 ^
  --lr 2e-4 ^
  --gate ^
  --gate-interval 10 ^
  --gate-games 8 ^
  --gate-sims 80 ^
  --gate-threshold 0.55 ^
  --baseline "%ROOT%checkpoints_v7_baseline\iter_200.pt" ^
  >> "%ROOT%logs\train_v7.log" 2>&1

echo [overnight] finished
