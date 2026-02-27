# Project Status: SinMDM - Dunhuang Dance Generation

**Last Updated**: 2026-02-27

## Current Objectives
1.  **Optimize Training**: Retrain SinMDM on Dunhuang dance data using official parameters to improve motion quality and stability.
2.  **Standardization**: Ensure exported BVH files follow industry standards for better compatibility with 3D software (Blender/Unity).
3.  **Synchronization**: Maintain project records and upload updates to GitHub regularly.

## Progress Summary
- **Environment**: Python 3.8 environment with SinMDM dependencies installed.
- **Data**: Dunhuang dance BVH/NPY files available.
- **BVH Standardization**: Implemented 3-channel joint export, Euler unwrap, and strict column count assertions (69 columns for 22 joints).
- **Rendering Optimization**: Implemented Auto-Framing (BBox limits) to ensure skeleton visibility in generated videos.
- **Generation**: Completed batch generation for `01-1-FeiTian` (20k) and `06-1-PiPaJiYue` (10k-30k).

## Active Experiments
### Experiment 1: Official Parameter Run
- **Status**: Completed for multiple models (20k-30k steps).
- **Config**: 
    - Arch: QnA
    - Scheduler: ExponentialLR
    - Gamma: 0.99998
    - Input: `bvh_general` dataset

## Todo List
- [x] Configure training command.
- [x] Create project documentation (walkthrough.md).
- [x] Standardize BVH Exporter (3-channel, unwrap).
- [x] Optimize Video Rendering (Auto-Framing).
- [x] Batch Generate `06-1-PiPaJiYue` Previews.
- [ ] Execute training/generation for `06-2-PiPaJiYue`.
- [ ] Verify generated motion quality in Blender.

## Key Files & Locations
- **Codebase**: `d:\sinMDM\sinmdm`
- **Documentation**: `d:\sinMDM\sinmdm\docs\`
- **Training Output**: `d:\sinMDM\sinmdm\save\`

## Notes/Issues
- BVH "torso monster" issue resolved via 3-channel standardization.
- Euler "jumping frames" resolved via unwrap logic.
- White-screen rendering issue resolved via auto-framing.
