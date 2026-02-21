# Project Status: SinMDM - Dunhuang Dance Generation

**Last Updated**: 2026-02-01

## Current Objectives
1.  **Optimize Training**: Retrain SinMDM on Dunhuang dance data using official parameters to improve motion quality and stability.
2.  **Paper Preparation**: Document training processes and results for upcoming publication.
3.  **Synchronization**: Maintain this status document to ensure continuity across development sessions.

## Progress Summary
- **Environment**: Python 3.8 environment with SinMDM dependencies installed.
- **Data**: Dunhuang dance BVH/NPY files available.
- **Current Task**: Initiating a new training run with `ExponentialLR` and official benchmark settings.

## Active Experiments
### Experiment 1: Official Parameter Run
- **Status**: Ready to Execute (User provided with one-liner command)
- **Log Path**: `C:\Users\JIANGJ~1\AppData\Local\Temp\openai-2026-02-01-16-02-01-127825`
- **Config**: 
    - Arch: QnA
    - Scheduler: ExponentialLR
    - Gamma: 0.99998
    - Steps: 20,000
    - Input: `d:/sinMDM/dataset/dunhuang/test_motion.npy`
    - Evaluators: Pending Manual Download
    - Generation: Enabled
- **Current Metric**: Step 1000 Loss: 6.54

## Technical Instructions
### Monitoring Training
As the background process is running, you can monitor the progress by checking the log file:
1. Open a new terminal.
2. Run: `Get-Content "C:\Users\JIANGJ~1\AppData\Local\Temp\openai-2026-02-01-16-02-01-127825\log.txt" -Wait` (if using PowerShell).

### Manual Evaluator Download
To enable `--eval_during_training` in future runs:
1. Download from: [Google Drive Link](https://drive.google.com/file/d/1O_GUHgjDbl2tgbyfSwZOUYXDACnk25Kb/view)
2. Save the file as `t2m.zip` in `d:\sinMDM\sinmdm\`.
3. Extract the contents. Ensure the final path exists: `d:\sinMDM\sinmdm\t2m\text_mot_match\model\finest.tar`.

## Todo List
- [x] Configure training command.
- [x] Create project documentation (walkthrough.md).
- [ ] Execute training run (User).
- [ ] Monitor loss and generate intermediate samples.
- [ ] Update `training_record.md` with results.
- [ ] Verify generated motion quality.

## Key Files & Locations
- **Codebase**: `d:\sinMDM\sinmdm`
- **Data**: `d:\sinMDM\dunhuang_dance_gen\data` (or similar)
- **Training Output**: `d:\sinMDM\sinmdm\save\`

## Notes/Issues
- Previous runs may have had stability issues; watching for NaN loss.
- Ensure GPU utilization is optimal.
