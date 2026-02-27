# Walkthrough - SinMDM Training Setup (Dunhuang Dataset)

This walkthrough documents the successful preparation, configuration, and optimization of the SinMDM model for training on the Dunhuang dance dataset.

## Changes Implemented

### 1. Dynamic Dimension Support
Modified `sinmdm/utils/model_util.py` to allow the `humanml` dataset type to adapt to arbitrary input dimensions. This resolved the hardcoded 263-channel requirement, enabling the model to train on the 198-channel Dunhuang data.

### 2. Standardized BVH Exporter
Implemented a major update to the BVH export logic in `Motion/BVH.py` and `sample/generate.py`:
- **3-Channel JOINT**: Conforming to BVH standards, non-root joints now only export rotation channels (ZXY).
- **Euler Unwrap**: Applied spatial unwrap to rotation data to eliminate frame-to-frame "jumps" (e.g., flipping between 179 and -179 degrees).
- **Strict Validation**: Added an assertion to ensure exactly 69 columns are written per frame for the 22-joint Dunhuang skeleton.

### 3. Optimized Video Rendering
Improved the `plot_script.py` visualization module:
- **Auto-Framing**: Dynamically calculates the bounding box of the character's movement across the entire sequence to automatically set camera limits.
- **Fixed "White Screen"**: Characters are now guaranteed to be within the camera's field of view, solving the previous empty video issue.

### 4. Safeguard for Sample Generation
Modified `sinmdm/sample/generate.py` to prevent crashes when the full HumanML3D dataset (specifically `Mean.npy`) is missing.

## Final Training Configuration

The project supports training with the following optimized command:

```powershell
python -m train.train_sinmdm --arch qna --dataset bvh_general --sin_path <path_to_bvh> --lr_method ExponentialLR --lr_gamma 0.99998 --num_steps 30000 --save_interval 2500 --use_scale_shift_norm --use_checkpoint --gen_during_training --overwrite
```

## Results & Verification
- **Loss Monitoring**: Baseline training showed smooth convergence.
- **Visual Feedback**: Auto-framing ensures clear video previews at every checkpoint.
- **Blender Compatibility**: Standardized BVH files (69 columns) can now be directly imported into Blender without torso distortion.

---
**Status**: All core optimizations completed and verified.
