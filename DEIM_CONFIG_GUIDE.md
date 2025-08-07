# DEIM 配置系统详解

## 配置继承层级

DEIM 使用了一个复杂的配置继承系统，通过 `__include__` 来组合多个配置文件：

```
你的配置文件
├── base/dataloader.yml      # 数据加载器基础配置
├── base/dfine_hgnetv2.yml   # D-FINE 模型架构配置
├── base/optimizer.yml       # 优化器配置
├── base/deim.yml           # DEIM 特定配置（增强版 D-FINE）
├── dataset/xxx.yml         # 数据集特定配置
└── runtime.yml             # 运行时配置（AMP、EMA等）
```

## 配置加载顺序很重要！

后面的配置会覆盖前面的配置。例如：
- `base/deim.yml` 会覆盖 `base/dfine_hgnetv2.yml` 中的某些设置
- 你的配置文件中的设置会覆盖所有包含文件的设置

## 常见配置陷阱

### 1. 数据集配置必须完整

```yaml
train_dataloader:
  dataset:
    type: CocoDetection  # 必须指定类型
    img_folder: /path/to/images
    ann_file: /path/to/annotations.json
    return_masks: False
    transforms:  # 必须有 transforms
      ops: ~  # 使用基础配置的 ops
      type: Compose
```

### 2. Transforms 配置结构

数据转换有两个层级：
- `transforms.ops`: 具体的转换操作列表
- `transforms.policy`: 控制哪些 ops 在哪个 epoch 停用

```yaml
transforms:
  ops:  # 如果设为 ~，会使用基础配置的默认 ops
    - {type: Resize, size: [640, 640]}
    - {type: ConvertPILImage, dtype: 'float32', scale: True}
    - # ... 其他转换
  policy:
    epoch: [4, 29, 50]  # 在这些 epoch 停用某些 ops
    ops: ['Mosaic', 'RandomPhotometricDistort']  # 要停用的 ops
```

### 3. HGNetv2 模型变体

不同大小的模型需要不同的配置：

| 模型 | name | 输出通道 | 编码器输入通道 |
|------|------|----------|----------------|
| Nano | B0   | [64, 256, 512, 1024] | [256, 512, 1024] |
| Small | B0  | [64, 256, 512, 1024] | [256, 512, 1024] |
| Medium | B2 | [64, 256, 512, 1024] | [256, 512, 1024] |
| Large | B4  | [128, 512, 1024, 2048] | [512, 1024, 2048] |
| XLarge | B5 | [128, 512, 1024, 2048] | [512, 1024, 2048] |

### 4. 关键配置参数

- `base_size`: 训练时的基础图像尺寸（影响位置编码）
- `eval_spatial_size`: 评估时的图像尺寸
- `total_batch_size`: 总批量大小（会自动分配到多个 GPU）
- `epoches` (注意拼写): 训练轮数

### 5. 文件包含顺序建议

```yaml
__include__:
  - ../base/dataloader.yml       # 基础数据加载
  - ../base/dfine_hgnetv2.yml   # 模型架构
  - ../base/optimizer.yml        # 优化器
  - ../base/deim.yml            # DEIM 增强
  - ../dataset/your_dataset.yml  # 你的数据集
  - ../runtime.yml              # 运行时设置
```

## 调试技巧

1. **检查最终配置**: 在 `train.py` 中打印 `cfg` 查看最终合并的配置
2. **逐步简化**: 先使用最少的配置，确保能运行，再逐步添加
3. **检查数据格式**: 确保数据集返回的是正确格式（图像应该被转换为张量）
4. **验证路径**: 使用绝对路径，避免相对路径问题

## 常见错误和解决方案

### 错误: 'Image' object is not subscriptable
**原因**: 数据没有经过正确的转换，仍然是 PIL Image
**解决**: 确保 transforms 配置正确，特别是包含 ConvertPILImage

### 错误: The size of tensor a (X) must match the size of tensor b (Y)
**原因**: 位置编码维度不匹配
**解决**: 检查 base_size 和模型配置是否匹配

### 错误: KeyError: '_pymodule'
**原因**: 配置系统无法找到模块定义
**解决**: 确保包含了所有必要的基础配置文件