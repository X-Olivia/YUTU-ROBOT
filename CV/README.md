# 🧻 Hygiene Products Detection and Tracking System

基于YOLOv8和Supervision的卫生用品检测、跟踪和计数系统。

## 📁 项目结构

```
CV/
├── requirements.txt          # 依赖包配置
├── config.py                # 配置文件
├── models/
│   └── detector.py  # 卫生用品检测模块
├── tracking/
│   └── tracker.py   # 卫生用品跟踪模块
├── utils/
│   └── video_processor.py   # 视频处理工具
├── main.py                  # 主运行文件
├── setup.sh                 # 环境配置脚本
├── runs/                    # 训练好的模型
│   └── detect/
│       └── train/
│           └── weights/
│               ├── best.pt  # 最佳权重
│               └── last.pt  # 最新权重
└── output/                  # 输出目录
```

## 🚀 快速开始

### 1. 环境配置

```bash
# 给脚本执行权限
chmod +x setup.sh

# 运行环境配置脚本
./setup.sh
```

### 2. 运行系统

```bash
# 运行完整的车辆检测和跟踪
python main.py

# 运行单帧测试
python main.py --test
```

## ⚙️ 配置说明

在 `config.py` 中可以调整以下参数：

- **模型设置**: 置信度阈值、IoU阈值
- **跟踪设置**: 跟踪激活阈值、丢失延迟
- **计数线设置**: 位置坐标
- **视频处理**: 帧跳过、输出帧率
- **可视化**: 框厚度、文本大小等

## 📊 输出结果

运行完成后，在 `output/` 目录下会生成：

1. **output_video.mp4**: 带有检测框、跟踪轨迹和计数信息的处理后的视频
2. **hygiene_stats.json**: 卫生用品统计信息（总数、分类统计等）

## 🔧 主要功能

- ✅ 卫生用品检测 (YOLOv8)
- ✅ 手部检测 (YOLOv8)
- ✅ 物品跟踪 (ByteTrack)
- ✅ 物品计数 (计数线)
- ✅ 轨迹可视化
- ✅ 实时统计显示
- ✅ 批量视频处理
