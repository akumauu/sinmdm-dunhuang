# Dunhuang Dance Motion Generation System
# 敦煌舞蹈动作生成系统

基于 SinMDM（Single Motion Diffusion）的敦煌舞蹈动作生成系统，实现从姿态序列到三维动作资产的完整流程。

## 项目结构

```
dunhuang_dance_gen/
├── data/               # 数据处理模块
│   ├── bvh_parser.py   # BVH 文件解析
│   ├── preprocess.py   # 预处理流程
│   └── validator.py    # 数据验证
├── models/             # 模型封装
│   └── sinmdm_wrapper.py
├── postprocess/        # 后处理模块
│   ├── smooth.py       # 动作平滑
│   └── constraints.py  # 物理约束
├── export/             # 导出模块
│   └── bvh_writer.py   # BVH 写入
├── visualize/          # 可视化模块
│   └── viewer.py       # 3D 查看器
└── scripts/            # 脚本入口
    └── demo.py         # 演示脚本
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r dunhuang_dance_gen/requirements.txt
```

### 2. 配置 SinMDM

```bash
cd sinmdm
conda env create -f environment.yml
conda activate SinMDM
```

### 3. 测试模块

```bash
python dunhuang_dance_gen/scripts/demo.py --test
```

### 4. 处理 BVH 文件

```bash
python dunhuang_dance_gen/scripts/demo.py --bvh dataset/dunhuang/example.bvh
```

## 核心功能

### 数据处理

```python
from dunhuang_dance_gen.data import load_bvh, DunhuangPreprocessor

# 加载 BVH
data = load_bvh("motion.bvh")

# 预处理
preprocessor = DunhuangPreprocessor(target_fps=30.0)
processed = preprocessor.process(data)
```

### 模型训练与生成

```python
from dunhuang_dance_gen.models import SinMDMWrapper

wrapper = SinMDMWrapper()

# 训练
model_path = wrapper.train("input.bvh", "output/models/")

# 生成
results = wrapper.generate(model_path, num_samples=5, motion_length=10.0)
```

### 后处理

```python
from dunhuang_dance_gen.postprocess import MotionSmoother, PhysicalConstraints

smoother = MotionSmoother(method='savgol')
constraints = PhysicalConstraints()

smoothed = smoother.smooth(positions)
final_pos, final_rot = constraints.apply_all(positions, rotations, joint_names)
```

### 导出与可视化

```python
from dunhuang_dance_gen.export import BVHWriter
from dunhuang_dance_gen.visualize import MotionViewer

# 导出
writer = BVHWriter()
writer.write("output.bvh", joint_names, parent_indices, offsets, positions, rotations)

# 可视化
viewer = MotionViewer()
viewer.animate_motion(joint_positions, parent_indices, save_path="animation.mp4")
```

## 许可证

MIT License
