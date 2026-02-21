# Walkthrough - SinMDM Training Setup (Dunhuang Dataset)

This walkthrough documents the successful preparation and configuration of the SinMDM model for training on the Dunhuang dance dataset using "official" parameters.

## Changes Implemented

### 1. Dynamic Dimension Support
Modified `sinmdm/utils/model_util.py` to allow the `humanml` dataset type to adapt to arbitrary input dimensions. This resolved the hardcoded 263-channel requirement, enabling the model to train on the 198-channel Dunhuang data.

```python
# sinmdm/utils/model_util.py
if args.dataset == 'humanml':
    data_rep = 'hml_vec'
    njoints = num_joints if num_joints is not None else 263
    nfeats = 1
```

### 2. Safeguard for Sample Generation
Modified `sinmdm/sample/generate.py` to prevent crashes when the full HumanML3D dataset (specifically `Mean.npy`) is missing. This is critical for SinMDM runs where only a single motion or a subset of data is provided.

*   Added checks for `Mean.npy` existence.
*   Handled `None` data objects during inverse transform.

### 3. Training Loop Optimization
Identified and resolved a directory mismatch issue (`ModuleNotFoundError`) by ensuring the correct execution context (`sinmdm` subdirectory).

## Final Training Configuration

The following command was provided to initiate the official 20,000-step training run:

```powershell
cd d:\sinMDM\sinmdm; python -m train.train_sinmdm --arch qna --dataset humanml --save_dir ./save/dunhuang_official_run --sin_path d:/sinMDM/dataset/dunhuang/test_motion.npy --lr_method ExponentialLR --lr_gamma 0.99998 --num_steps 20000 --save_interval 2000 --use_scale_shift_norm --use_checkpoint --gen_during_training --overwrite
```

## Results & Verification
- **Loss Monitoring**: Baseline training showed loss decreasing from ~0.27 towards stability.
- **Visual Feedback**: Enabled `--gen_during_training` to provide periodic motion samples in `./save/dunhuang_official_run`.
- **Checkpointing**: Every 2,000 steps, a model checkpoint is saved for later fine-tuning or synthesis.

---
**Status**: Ready for Execution.
