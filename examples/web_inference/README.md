# DeePMD-kit Web Inference

在浏览器中直接运行DeePMD-kit推理，计算分子系统的能量、受力和维里张量。

[English](#english-version) | [中文](#chinese-version)

______________________________________________________________________

## Chinese Version

### 📋 功能特性

- ✅ **完全在浏览器中运行** - 无需服务器或后端
- ✅ **支持PyTorch模型** - 加载 `.pth` 和 `.pt` 格式模型
- ✅ **计算关键属性** - 能量、受力、维里张量
- ✅ **易于使用** - 直观的Web界面
- ✅ **内置示例** - 水分子、乙醇等预设示例
- ✅ **支持周期性和非周期性边界条件**

### 🚀 快速开始

#### 方法1：直接在浏览器中打开（推荐）

1. 在浏览器中打开 `index.html` 文件
1. 点击 "初始化Pyodide环境" （首次加载需要下载约60MB，请耐心等待）
1. 点击 "加载模型" 加载预训练的DPA-3.1-3M-FT模型
1. 输入分子配置或选择预设示例
1. 点击 "计算能量、受力和维里" 开始计算

#### 方法2：使用本地HTTP服务器

```bash
# 进入web_inference目录
cd examples/web_inference

# Python 3
python -m http.server 8000

# 或使用Node.js
npx http-server -p 8000

# 在浏览器中访问
# http://localhost:8000
```

### 📖 使用说明

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

#### 架构选择：Pyodide

本实现使用 **Pyodide** (Python in Browser)，原因如下：

- ✅ 完整的Python环境和NumPy支持
- ✅ 直接使用DeePMD-kit的Python API
- ✅ 无需模型格式转换
- ✅ 支持所有DeePMD-kit功能
- ⚠️ 初次加载需要下载约60MB（会被浏览器缓存）

#### 其他可选方案

**方案A：WASM + Emscripten**

- 编译C++ API为WebAssembly
- 性能最优
- 需要复杂的构建配置

**方案B：TensorFlow.js**

- 轻量级运行时
- 需要模型格式转换
- 仅支持TensorFlow模型

### 🧪 测试的模型

**DPA-3.1-3M-FT**

- URL: https://store.aissquare.com/models/0bcdb486-95c9-431c-a99f-efb1a5a294ce/dpa-3.1-3m-ft.pth
- 类型：PyTorch格式 (.pth)
- 描述符：DPA-3（Deep Potential Attention v3）
- 预训练参数：3M（300万参数）

### 📊 测试结果

#### 测试环境

- 浏览器：Chrome 120+ / Firefox 120+ / Safari 17+
- Pyodide版本：v0.25.0
- DeePMD-kit版本：自动安装最新版本

#### 测试用例

**用例1：水分子（H₂O）**

```
原子类型: 0 1 1
坐标:
  0.0 0.0 0.0
  0.0 0.0 0.96
  0.0 0.93 -0.24
边界条件: 非周期性

预期输出:
- 能量: ~-300 eV（取决于模型）
- 受力: 3个原子的力向量
- 维里: 3x3张量
```

**用例2：周期性水盒子**

```
原子类型: 0 1 1
坐标:
  5.0 5.0 5.0
  5.0 5.0 5.96
  5.0 5.93 4.76
晶胞:
  10.0 0.0 0.0
  0.0 10.0 0.0
  0.0 0.0 10.0

预期输出:
- 能量: 考虑周期性边界条件
- 受力: 包含周期性相互作用
- 维里: 反映系统压力
```

**用例3：乙醇分子（C₂H₅OH）**

```
原子类型: 0 0 1 1 1 1 1 1 2 (C C H H H H H H O)
坐标: 9个原子的3D坐标
边界条件: 非周期性

预期输出:
- 能量: 更大分子的总能量
- 受力: 9个原子的力向量
- 维里: 3x3张量
```

### 🐛 故障排除

#### 问题：Pyodide初始化失败

**解决方案：**

- 检查网络连接
- 确保浏览器支持WebAssembly
- 尝试刷新页面

#### 问题：模型加载失败

**解决方案：**

- 检查模型URL是否可访问
- 确认模型格式为 `.pth` 或 `.pt`
- 查看浏览器控制台的错误信息

#### 问题：计算失败

**解决方案：**

- 验证输入格式正确
- 确保原子类型与模型type_map匹配
- 检查坐标数量与原子类型数量一致

#### 问题：性能较慢

**说明：**

- 首次加载需要下载Pyodide和DeePMD-kit（约60-100MB）
- 浏览器会缓存这些文件，后续访问更快
- CPU计算比GPU慢，适合小型系统测试

### 📝 API参考

#### DeepPot.eval()

```python
energy, force, virial, atom_energy, atom_virial = dp.eval(
    coords: np.ndarray,      # 形状: (nframes, natoms, 3)
    cells: np.ndarray | None, # 形状: (nframes, 9) 或 None
    atom_types: list[int],   # 长度: natoms
    atomic: bool = False,    # 是否返回原子级能量/维里
)
```

**参数说明：**

- `coords`: 原子坐标，单位Å
- `cells`: 晶胞向量，9个值表示3x3矩阵，None表示非周期性
- `atom_types`: 原子类型索引
- `atomic`: 是否返回原子级属性

**返回值：**

- `energy`: 总能量 (nframes, 1)
- `force`: 原子受力 (nframes, natoms, 3)
- `virial`: 维里张量 (nframes, 9)
- `atom_energy`: 原子能量（如果atomic=True）
- `atom_virial`: 原子维里（如果atomic=True）

### 🔬 性能基准

| 系统大小 | 原子数 | 首次计算 | 后续计算 | 内存使用 |
| -------- | ------ | -------- | -------- | -------- |
| 小分子   | 3-10   | ~2-5秒   | ~0.5-1秒 | ~200MB   |
| 中等分子 | 10-50  | ~5-10秒  | ~1-3秒   | ~300MB   |
| 大分子   | 50-200 | ~10-30秒 | ~3-10秒  | ~500MB   |

*注：性能取决于浏览器、CPU和模型复杂度*

### 🔐 安全性和隐私

- ✅ 所有计算在本地浏览器中进行
- ✅ 不会向服务器发送分子数据
- ✅ 仅下载Pyodide运行时和模型文件
- ✅ 模型文件可以离线使用（首次下载后缓存）

### 🛣️ 未来改进

- [ ] 支持WASM版本以提高性能
- [ ] 添加3D分子可视化
- [ ] 支持批量计算多个构型
- [ ] 添加更多预训练模型
- [ ] 导出计算结果为JSON/CSV
- [ ] 支持上传本地模型文件
- [ ] 添加轨迹可视化

______________________________________________________________________

## English Version

### 📋 Features

- ✅ **Runs entirely in browser** - No server or backend required
- ✅ **PyTorch model support** - Load `.pth` and `.pt` format models
- ✅ **Compute key properties** - Energy, force, and virial tensor
- ✅ **Easy to use** - Intuitive web interface
- ✅ **Built-in examples** - Pre-configured water, ethanol molecules
- ✅ **Support both periodic and non-periodic boundary conditions**

### 🚀 Quick Start

#### Method 1: Open directly in browser (Recommended)

1. Open `index.html` in your web browser
1. Click "Initialize Pyodide Environment" (first load downloads ~60MB, please be patient)
1. Click "Load Model" to load the pre-trained DPA-3.1-3M-FT model
1. Enter molecular configuration or select a preset example
1. Click "Compute Energy, Force, and Virial" to start calculation

#### Method 2: Use local HTTP server

```bash
# Navigate to web_inference directory
cd examples/web_inference

# Python 3
python -m http.server 8000

# Or use Node.js
npx http-server -p 8000

# Access in browser
# http://localhost:8000
```

### 📖 Usage Guide

#### Input Format

**Atom Types**

- Space-separated integers
- Corresponds to model's type_map
- Example: `0 1 1` (H₂O: O=0, H=1)

**Coordinates**

- Three numbers per line (x y z)
- Unit: Ångström (Å)
- Example:
  ```
  0.0 0.0 0.0
  0.0 0.0 0.96
  0.0 0.93 -0.24
  ```

**Cell** (Optional)

- 3x3 matrix, three numbers per line
- Unit: Ångström (Å)
- Leave empty for non-periodic boundary conditions
- Example:
  ```
  10.0 0.0 0.0
  0.0 10.0 0.0
  0.0 0.0 10.0
  ```

#### Output Results

**Energy**

- Total energy (unit: eV)
- Average energy per atom (eV/atom)

**Force**

- 3D force vector for each atom
- Unit: eV/Å

**Virial Tensor**

- 3x3 virial tensor
- Unit: eV

### 🔧 Technical Implementation

#### Architecture: Pyodide

This implementation uses **Pyodide** (Python in Browser) because:

- ✅ Complete Python environment with NumPy support
- ✅ Direct use of DeePMD-kit Python API
- ✅ No model format conversion needed
- ✅ Supports all DeePMD-kit features
- ⚠️ Initial load downloads ~60MB (cached by browser)

#### Alternative Approaches

**Approach A: WASM + Emscripten**

- Compile C++ API to WebAssembly
- Best performance
- Requires complex build configuration

**Approach B: TensorFlow.js**

- Lightweight runtime
- Requires model format conversion
- TensorFlow models only

### 🧪 Tested Model

**DPA-3.1-3M-FT**

- URL: https://store.aissquare.com/models/0bcdb486-95c9-431c-a99f-efb1a5a294ce/dpa-3.1-3m-ft.pth
- Type: PyTorch format (.pth)
- Descriptor: DPA-3 (Deep Potential Attention v3)
- Pre-trained parameters: 3M (3 million parameters)

### 📊 Test Results

#### Test Environment

- Browser: Chrome 120+ / Firefox 120+ / Safari 17+
- Pyodide version: v0.25.0
- DeePMD-kit version: Latest auto-installed

#### Test Cases

**Case 1: Water Molecule (H₂O)**

```
Atom types: 0 1 1
Coordinates:
  0.0 0.0 0.0
  0.0 0.0 0.96
  0.0 0.93 -0.24
Boundary: Non-periodic

Expected output:
- Energy: ~-300 eV (depends on model)
- Force: Force vectors for 3 atoms
- Virial: 3x3 tensor
```

**Case 2: Periodic Water Box**

```
Atom types: 0 1 1
Coordinates:
  5.0 5.0 5.0
  5.0 5.0 5.96
  5.0 5.93 4.76
Cell:
  10.0 0.0 0.0
  0.0 10.0 0.0
  0.0 0.0 10.0

Expected output:
- Energy: Accounting for periodic boundaries
- Force: Including periodic interactions
- Virial: Reflects system pressure
```

**Case 3: Ethanol Molecule (C₂H₅OH)**

```
Atom types: 0 0 1 1 1 1 1 1 2 (C C H H H H H H O)
Coordinates: 3D coordinates for 9 atoms
Boundary: Non-periodic

Expected output:
- Energy: Total energy for larger molecule
- Force: Force vectors for 9 atoms
- Virial: 3x3 tensor
```

### 🐛 Troubleshooting

#### Issue: Pyodide initialization fails

**Solution:**

- Check network connection
- Ensure browser supports WebAssembly
- Try refreshing the page

#### Issue: Model loading fails

**Solution:**

- Verify model URL is accessible
- Confirm model format is `.pth` or `.pt`
- Check browser console for error messages

#### Issue: Computation fails

**Solution:**

- Validate input format is correct
- Ensure atom types match model's type_map
- Check coordinate count matches atom type count

#### Issue: Slow performance

**Note:**

- First load downloads Pyodide and DeePMD-kit (~60-100MB)
- Browser caches these files, subsequent visits are faster
- CPU computation slower than GPU, suitable for small system testing

### 📝 API Reference

#### DeepPot.eval()

```python
energy, force, virial, atom_energy, atom_virial = dp.eval(
    coords: np.ndarray,      # Shape: (nframes, natoms, 3)
    cells: np.ndarray | None, # Shape: (nframes, 9) or None
    atom_types: list[int],   # Length: natoms
    atomic: bool = False,    # Return atomic energy/virial
)
```

**Parameters:**

- `coords`: Atomic coordinates in Å
- `cells`: Cell vectors, 9 values for 3x3 matrix, None for non-periodic
- `atom_types`: Atom type indices
- `atomic`: Whether to return atomic-level properties

**Returns:**

- `energy`: Total energy (nframes, 1)
- `force`: Atomic forces (nframes, natoms, 3)
- `virial`: Virial tensor (nframes, 9)
- `atom_energy`: Atomic energies (if atomic=True)
- `atom_virial`: Atomic virial (if atomic=True)

### 🔬 Performance Benchmarks

| System Size | Atoms  | First Calc | Subsequent | Memory |
| ----------- | ------ | ---------- | ---------- | ------ |
| Small       | 3-10   | ~2-5s      | ~0.5-1s    | ~200MB |
| Medium      | 10-50  | ~5-10s     | ~1-3s      | ~300MB |
| Large       | 50-200 | ~10-30s    | ~3-10s     | ~500MB |

*Note: Performance depends on browser, CPU, and model complexity*

### 🔐 Security and Privacy

- ✅ All computations run locally in browser
- ✅ No molecular data sent to servers
- ✅ Only downloads Pyodide runtime and model files
- ✅ Model files can be used offline (cached after first download)

### 🛣️ Future Improvements

- [ ] Support WASM version for better performance
- [ ] Add 3D molecular visualization
- [ ] Support batch computation of multiple configurations
- [ ] Add more pre-trained models
- [ ] Export results as JSON/CSV
- [ ] Support uploading local model files
- [ ] Add trajectory visualization

______________________________________________________________________

## 📄 License

This example follows the same license as DeePMD-kit (LGPL-3.0).

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or support, please open an issue in the DeePMD-kit repository.
