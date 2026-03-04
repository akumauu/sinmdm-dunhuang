# SinMDM Training Record - Dunhuang Dance

## Training Run: `01-1-FeiTian`
**Date**: 2026-02-27
**Dataset**: `D:\sinMDM\敦煌舞三维动作数据集\长动作\01-FeiTian\01-1-FeiTian\01-1-FeiTian.bvh`
**Parameters**:
- `arch`: qna
- `dataset`: bvh_general
- `num_steps`: 30,000
- `save_interval`: 2,500
- `lr_method`: ExponentialLR
- `lr_gamma`: 0.99998

## Results Summary
| Step | Status | Motion Quality |
|------|--------|----------------|
| 5000 | Good | Stable transitions |
| 20000| Optimal| High variety, smooth |
| 30000| Final | Completed |

---

## Training Run: `06-1-PiPaJiYue`
**Date**: 2026-02-27
**Dataset**: `D:\sinMDM\敦煌舞三维动作数据集\长动作\06-PiPaJiYue\06-1-PiPaJiYue\06-1-PiPaJiYue.bvh`
**Parameters**:
- `arch`: qna
- `dataset`: bvh_general
- `num_steps`: 30,000
- `save_interval`: 2,500
- `lr_method`: ExponentialLR
- `lr_gamma`: 0.99998

## Results Summary
| Step | Status | Motion Quality |
|------|--------|----------------|
| 10000| Good | Standard BVH format generated |
| 20000| Optimal| Clear skeletons with Auto-Framing |
| 29999| Final | Batch preview completed |

## Key Findings
1. **Standardization**: All generated samples now use the 3-channel JOINT format (69 columns total), fixing previous torso misalignment issues.
2. **Auto-Framing**: Video rendering now dynamically centers the character, resolving visibility issues.
3. **Euler Unwrap**: Rotation jumps are eliminated in exported BVH files.
