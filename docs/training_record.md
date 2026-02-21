# SinMDM Training Record - Dunhuang Dance

## Training Run: `feitian_huishen`
**Date**: 2026-02-01
**Dataset**: `D:\sinMDM\敦煌舞三维动作数据集\基础动作\1-FeiTian\01-HuiShenCeTuoShi\01-HuiShenCeTuoShi.bvh`
**Parameters**:
- `arch`: qna
- `dataset`: bvh_general
- `num_steps`: 20,000
- `save_interval`: 2000
- `lr_method`: ExponentialLR
- `lr_gamma`: 0.99998

## Results Summary
| Step | Loss | Motion Quality |
|------|------|----------------|
| 0 | 91.65 | Noise |
| 2000 | 2.89 | Good basic motion style |
| 6000 | ~2.5 | **Optimal** - Smooth, good variety |
| 8000 | ~2.3 | Good, slight overfitting signs |
| 19999| 1.95 | **Overfitted** - Shake/Jitter/Invalid Rotations |

## Key Findings
1. **Dimension Mismatch Solved**: Successfully implemented `bvh_general` mode which bypasses the 263-dim HumanML limit.
2. **Overfitting**: The model overfits significantly after step 10,000 due to the small dataset size (single motion file).
3. **Best Model**: `model000006000.pt` is recommended for generation.

## Visualizations
MP4 samples for step 2000-19999 have been generated. Step 12000+ samples show visual artifacts (empty/static) due to `divide by zero` errors in quaternion normalization, confirming overfitting.

---

## Training Run: `feitian_24s_v1`
**Date**: 2026-02-21
**Dataset**: `D:\sinMDM\敦煌舞三维动作数据集\长动作\06-PiPaJiYue\06-1-PiPaJiYue\06-1-PiPaJiYue.bvh`
**Parameters**:
- `arch`: qna
- `dataset`: bvh_general
- `batch_size`: 16 (from 64)
- `eval_batch_size`: 16 (from 32)
- `num_frames`: 90 (expanded to 3 seconds window)
- `dropout`: 0.3 (from 0.5)
- `drop_path`: 0.2 (from 0.5)
- `num_steps`: 20,000
- `save_interval`: 4000
- `lr_method`: ExponentialLR
- `lr_gamma`: 0.99998

## Results Summary
| Step | Loss | Motion Quality |
|------|------|----------------|
| 0 | 125.34 | Initial |
| 19999 | - | Training completed successfully |

## Key Findings
1. Models trained on longer 90-frame windows completed smoothly.
2. `batch_size` 16 / `eval_batch_size` 16 avoided out-of-memory errors on the larger frames.
3. Checkpoints and generated MP4 samples saved every 4000 steps up to 19999.
