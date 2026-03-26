# DeePMD-kit Web Inference (ONNX Runtime)

在浏览器中直接运行DeePMD-kit推理，计算分子系统的能量、受力和维里张量。

⚠️ **重要**: PyTorch不支持Pyodide（浏览器中的Python环境）。本实现使用**ONNX Runtime Web**进行浏览器端推理。

[English](#english-version) | [中文](#chinese-version)

______________________________________________________________________

## Chinese Version

### 📋 功能特性

- ✅ **完全在浏览器中运行** - 无需服务器或后端
- ✅ **使用ONNX Runtime Web** - 专为浏览器ML推理设计
- ✅ **计算关键属性** - 能量、受力、维里张量
- ✅ **易于使用** - 直观的Web界面
- ✅ **内置示例** - 水分子、乙醇等预设示例
- ✅ **支持周期性和非周期性边界条件**

### 🚀 快速开始

#### 步骤1：导出模型为ONNX格式

首先，您需要将DeePMD-kit PyTorch模型导出为ONNX格式：

```bash
cd examples/web_inference

# 导出模型
python export_to_onnx.py /path/to/your/model.pth -o model.onnx

# 或指定原子数量
python export_to_onnx.py model.pth -o model.onnx --natoms 3
```

**注意事项：**

- 需要安装DeePMD-kit PyTorch版本：`pip install deepmd-kit[torch]`
- 导出过程需要ONNX包：`pip install onnx`
- 并非所有模型架构都能成功导出为ONNX格式
- 导出的模型会针对特定原子数进行优化

#### 步骤2：在浏览器中使用

```bash
# 使用本地HTTP服务器
python -m http.server 8000

# 在浏览器中打开
# http://localhost:8000/index_onnx.html
```

#### 步骤3：加载和计算

1. 在界面中输入ONNX模型路径（例如：`model.onnx`）
1. 点击"加载ONNX模型"
1. 输入分子配置或选择预设示例
1. 点击"计算能量、受力和维里"

### 📖 使用说明

#### 模型导出

**基本用法：**

```bash
python export_to_onnx.py <model_path> [-o output.onnx] [--natoms N]
```

**参数说明：**

- `model_path`: 输入模型文件路径（.pt或.pth）
- `-o, --output`: 输出ONNX文件路径（默认: model.onnx）
- `-n, --natoms`: 导出时使用的原子数（默认: 3）

**示例：**

```bash
# 导出水分子模型（3个原子）
python export_to_onnx.py water_model.pth -o water.onnx

# 导出更大分子的模型
python export_to_onnx.py large_model.pth -o large.onnx --natoms 100
```

#### 输入格式

**原子类型（Atom Types）**

- 空格分隔的整数列表
- 对应模型的type_map映射
- 示例：`0 1 1` (H₂O: O=0, H=1)

**坐标（Coordinates）**

- 每行3个数字（x y z）
- 单位：埃（Å）
- 示例：
  ```
  0.0 0.0 0.0
  0.0 0.0 0.96
  0.0 0.93 -0.24
  ```

**晶胞参数（Cell）**（可选）

- 3x3矩阵，每行3个数字
- 单位：埃（Å）
- 留空表示非周期性边界条件
- 示例：
  ```
  10.0 0.0 0.0
  0.0 10.0 0.0
  0.0 0.0 10.0
  ```

#### 输出结果

**能量（Energy）**

- 总能量（单位：eV）
- 平均能量/原子（eV/atom）

**受力（Force）**

- 每个原子的三维力向量
- 单位：eV/Å

**维里张量（Virial）**

- 3x3维里张量
- 单位：eV

### 🔧 技术实现

#### 架构选择：ONNX Runtime Web

本实现使用 **ONNX Runtime Web**，原因如下：

- ✅ 专为浏览器ML推理设计
- ✅ 支持WebAssembly和WebGL加速
- ✅ 跨平台跨框架
- ✅ 体积相对较小（~2-5MB）
- ⚠️ 需要将模型转换为ONNX格式

#### 为什么不使用Pyodide？

**Pyodide限制：**

- ❌ PyTorch没有纯Python wheel包
- ❌ 无法在Pyodide中安装PyTorch
- ❌ TensorFlow同样不支持Pyodide
- ❌ 首次加载需要60-100MB

**ONNX Runtime Web优势：**

- ✅ 轻量级（~2-5MB）
- ✅ 加载速度快
- ✅ 支持硬件加速
- ✅ 专为浏览器优化

### 🧪 测试模型

您可以使用任何DeePMD-kit PyTorch模型，只需先导出为ONNX格式。

**示例：导出DPA-3.1-3M-FT模型**

```bash
# 1. 下载模型
wget https://store.aissquare.com/models/0bcdb486-95c9-431c-a99f-efb1a5a294ce/dpa-3.1-3m-ft.pth

# 2. 导出为ONNX
python export_to_onnx.py dpa-3.1-3m-ft.pth -o dpa-3m.onnx --natoms 3

# 3. 在浏览器中使用
# 打开 index_onnx.html 并加载 dpa-3m.onnx
```

### 📊 性能对比

| 方案             | 首次加载 | 推理速度 | 体积     | PyTorch支持 | 实用性      |
| ---------------- | -------- | -------- | -------- | ----------- | ----------- |
| Pyodide          | ~30-60s  | 慢       | 60-100MB | ❌ 不支持   | ❌ 无法使用 |
| ONNX Runtime Web | ~2-5s    | 快       | 2-5MB    | ✅ 通过导出 | ✅ 推荐使用 |
| WASM (C++)       | ~5-10s   | 很快     | 10-20MB  | ❌ 需要编译 | ⚠️ 实现复杂 |

### 🐛 故障排除

#### 问题：模型导出失败

**错误信息：** `Export failed: ... not supported in ONNX`

**解决方案：**

- 某些DeePMD模型架构可能使用了ONNX不支持的操作
- 尝试使用不同的模型或简化模型架构
- 检查是否使用了自定义算子

#### 问题：ONNX模型加载失败

**解决方案：**

- 确认模型文件路径正确
- 检查浏览器控制台是否有CORS错误
- 确保使用HTTP服务器（不要直接用file://协议）

#### 问题：计算结果不准确

**解决方案：**

- 确认原子类型与模型训练时的type_map一致
- 检查坐标单位（应为埃Å）
- 验证晶胞参数格式正确

#### 问题：浏览器不支持

**最低要求：**

- Chrome 90+
- Firefox 88+
- Safari 15+
- Edge 90+
- 需要支持WebAssembly

### 📝 API参考

#### export_to_onnx.py

```python
python export_to_onnx.py <model_path> [options]

参数:
  model_path          输入模型文件 (.pt 或 .pth)
  -o, --output PATH   输出ONNX文件路径 (默认: model.onnx)
  -n, --natoms N      示例原子数 (默认: 3)

示例:
  python export_to_onnx.py model.pth -o output.onnx
  python export_to_onnx.py model.pth --natoms 10
```

#### ONNX Runtime Web API

```javascript
// 加载模型
const session = await ort.InferenceSession.create('model.onnx');

// 准备输入
const feeds = {
    coord: new ort.Tensor('float64', coordData, [nframes, natoms, 3]),
    atype: new ort.Tensor('int32', atypeData, [natoms]),
    box: new ort.Tensor('float64', boxData, [nframes, 3, 3])
};

// 运行推理
const results = await session.run(feeds);

// 获取输出
const energy = results.energy.data;
const force = results.force.data;
const virial = results.virial.data;
```

### 🔬 性能基准

| 系统大小 | 原子数 | ONNX加载 | 首次计算  | 后续计算   |
| -------- | ------ | -------- | --------- | ---------- |
| 小分子   | 3-10   | ~0.5-1s  | ~0.1-0.5s | ~0.05-0.2s |
| 中等分子 | 10-50  | ~0.5-1s  | ~0.5-2s   | ~0.2-1s    |
| 大分子   | 50-200 | ~1-2s    | ~2-5s     | ~1-3s      |

*注：性能取决于浏览器、设备和模型复杂度*

### 🔐 安全性和隐私

- ✅ 所有计算在本地浏览器中进行
- ✅ 不会向服务器发送分子数据
- ✅ 仅下载ONNX Runtime和模型文件
- ✅ 模型文件可以离线使用

### 🛣️ 未来改进

- [ ] 支持批量计算多个构型
- [ ] 添加3D分子可视化
- [ ] 支持更多模型格式
- [ ] WebGL加速支持
- [ ] 导出计算结果为JSON/CSV
- [ ] 支持拖放上传模型文件

______________________________________________________________________

## English Version

### 📋 Features

- ✅ **Runs entirely in browser** - No server or backend required
- ✅ **Uses ONNX Runtime Web** - Designed specifically for browser ML inference
- ✅ **Compute key properties** - Energy, force, and virial tensor
- ✅ **Easy to use** - Intuitive web interface
- ✅ **Built-in examples** - Pre-configured water, ethanol molecules
- ✅ **Support both periodic and non-periodic boundary conditions**

### 🚀 Quick Start

#### Step 1: Export Model to ONNX Format

First, export your DeePMD-kit PyTorch model to ONNX format:

```bash
cd examples/web_inference

# Export model
python export_to_onnx.py /path/to/your/model.pth -o model.onnx

# Or specify number of atoms
python export_to_onnx.py model.pth -o model.onnx --natoms 3
```

**Requirements:**

- DeePMD-kit with PyTorch: `pip install deepmd-kit[torch]`
- ONNX package: `pip install onnx`
- Not all model architectures can be exported to ONNX
- Exported model is optimized for specific atom count

#### Step 2: Use in Browser

```bash
# Start local HTTP server
python -m http.server 8000

# Open in browser
# http://localhost:8000/index_onnx.html
```

#### Step 3: Load and Compute

1. Enter ONNX model path in the interface (e.g., `model.onnx`)
1. Click "Load ONNX Model"
1. Enter molecular configuration or select preset example
1. Click "Compute Energy, Force, and Virial"

### 📖 Usage Guide

#### Model Export

**Basic Usage:**

```bash
python export_to_onnx.py <model_path> [-o output.onnx] [--natoms N]
```

**Arguments:**

- `model_path`: Input model file path (.pt or .pth)
- `-o, --output`: Output ONNX file path (default: model.onnx)
- `-n, --natoms`: Number of atoms for export (default: 3)

**Examples:**

```bash
# Export water molecule model (3 atoms)
python export_to_onnx.py water_model.pth -o water.onnx

# Export larger molecule model
python export_to_onnx.py large_model.pth -o large.onnx --natoms 100
```

### 🔧 Technical Implementation

#### Why ONNX Runtime Web?

**Pyodide Limitations:**

- ❌ PyTorch has no pure Python wheel
- ❌ Cannot install PyTorch in Pyodide
- ❌ TensorFlow also unsupported in Pyodide
- ❌ Initial load requires 60-100MB

**ONNX Runtime Web Advantages:**

- ✅ Lightweight (~2-5MB)
- ✅ Fast loading
- ✅ Hardware acceleration support
- ✅ Optimized for browsers

### 📊 Performance Comparison

| Approach         | Initial Load | Inference Speed | Size     | PyTorch Support      | Practicality   |
| ---------------- | ------------ | --------------- | -------- | -------------------- | -------------- |
| Pyodide          | ~30-60s      | Slow            | 60-100MB | ❌ Not supported     | ❌ Cannot use  |
| ONNX Runtime Web | ~2-5s        | Fast            | 2-5MB    | ✅ Via export        | ✅ Recommended |
| WASM (C++)       | ~5-10s       | Very fast       | 10-20MB  | ❌ Needs compilation | ⚠️ Complex     |

### 🐛 Troubleshooting

#### Issue: Model export fails

**Error:** `Export failed: ... not supported in ONNX`

**Solution:**

- Some DeePMD model architectures use operations not supported in ONNX
- Try a different model or simplify model architecture
- Check for custom operators

#### Issue: ONNX model loading fails

**Solution:**

- Verify model file path is correct
- Check browser console for CORS errors
- Ensure using HTTP server (don't use file:// protocol)

#### Issue: Inaccurate results

**Solution:**

- Confirm atom types match model's type_map
- Check coordinate units (should be Ångström)
- Verify cell parameters format

### 🔬 Performance Benchmarks

| System Size | Atoms  | ONNX Load | First Calc | Subsequent |
| ----------- | ------ | --------- | ---------- | ---------- |
| Small       | 3-10   | ~0.5-1s   | ~0.1-0.5s  | ~0.05-0.2s |
| Medium      | 10-50  | ~0.5-1s   | ~0.5-2s    | ~0.2-1s    |
| Large       | 50-200 | ~1-2s     | ~2-5s      | ~1-3s      |

*Note: Performance depends on browser, device, and model complexity*

______________________________________________________________________

## 📄 License

This example follows the same license as DeePMD-kit (LGPL-3.0).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or support, please open an issue in the DeePMD-kit repository.
