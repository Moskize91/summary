# Visualization Changes - Summary

## What Changed

按照方案B，我们合并了基础可视化和蛇检测可视化：

### Before
- `scripts/visualize.py` → 生成基础可视化（蓝色节点）
- `scripts/detect_snakes.py` → 检测蛇并生成彩色可视化

### After (方案B)
- `scripts/visualize.py` → **自动检测蛇并生成彩色可视化**
- 删除了 `scripts/detect_snakes.py`（功能已合并）

## Current Workflow

```bash
# 1. 提取知识图谱
python scripts/main.py

# 2. 可视化（自动包含蛇检测）
python scripts/visualize.py
```

生成的文件：
- `output/knowledge_graph.json` - 知识图谱数据
- `output/snakes.json` - 检测到的蛇
- `output/knowledge_graph.html` - 彩色可视化（包含蛇标注）

## Benefits of Plan B

✅ **单一入口**：只需运行 `visualize.py` 就能得到完整的可视化
✅ **信息丰富**：同时显示图谱结构和主题线索
✅ **无冗余**：不再有两个相似的可视化文件
✅ **灰色节点**：不属于任何蛇的节点会显示为灰色，仍然可见

## Deprecated Files

- `dev/visualizer.py` - 标记为 DEPRECATED（但保留以防有引用）
- `scripts/detect_snakes.py` - 已删除（功能合并到 `visualize.py`）

## Visualization Features

现在的 `knowledge_graph.html` 包含：
- 🎨 每条蛇用不同颜色标记
- 🔗 同一条蛇内的边用相应颜色加粗
- 📊 顶部图例显示所有检测到的蛇
- 💡 鼠标悬停显示节点详情和所属蛇编号
- 🌫️ 不属于任何蛇的节点显示为灰色

## Parameters

可以在 `scripts/visualize.py` 中调整蛇检测参数：

```python
detector = SnakeDetector(
    max_hops=3,              # 墨水扩散距离
    distance_threshold=0.5,   # 聚类距离阈值
    min_cluster_size=3        # 最小蛇长度
)
```
