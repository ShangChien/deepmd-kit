# Implementation Notes

## 实现方案 (Implementation Approaches)

本项目提供了在浏览器中运行DeePMD推理的解决方案。以下是已实现和可选的方案：

### ✅ 已实现：Pyodide方案 (Implemented: Pyodide Solution)

**文件：**

- `index.html` - 主Web界面
- `deepmd_inference.js` - JavaScript接口
- `test_inference.py` - 测试脚本

**优点：**

- ✅ 完整的Python环境和DeePMD-kit支持
- ✅ 无需模型格式转换
- ✅ 直接使用官方Python API
- ✅ 支持所有功能（能量、受力、维里）

**缺点：**

- ⚠️ 首次加载需要下载约60-80MB（Pyodide + NumPy + Torch + DeePMD-kit）
- ⚠️ CPU性能较慢（适合小分子测试）

**使用场景：**

- 教育和演示
- 小分子测试（< 100原子）
- 需要完整DeePMD-kit功能

______________________________________________________________________

### 🔄 备选方案1：WASM (Alternative: WebAssembly)

**状态：** 未实现，但可行

**实现步骤：**

1. 使用Emscripten编译C++ API (`source/api_cc/`)
1. 导出关键函数（`DeepPot::init`, `DeepPot::computew`）
1. 创建JavaScript绑定
1. 支持直接加载.pth/.pt模型

**优点：**

- ✅ 性能最优（接近原生速度）
- ✅ 体积较小（约10-20MB）
- ✅ 支持所有后端（TF, PyTorch, JAX）

**缺点：**

- ⚠️ 需要复杂的构建配置
- ⚠️ 需要编译PyTorch/TensorFlow为WASM
- ⚠️ 开发成本高

**参考：**

- `source/api_cc/src/DeepPot.cc` - C++ API实现
- `source/api_c/include/c_api.h` - C API（WASM友好）

______________________________________________________________________

### 🔄 备选方案2：TensorFlow.js (Alternative: TensorFlow.js)

**状态：** 未实现，需要模型转换

**实现步骤：**

1. 将PyTorch模型转换为TensorFlow SavedModel格式
1. 使用`tensorflowjs_converter`转换为TF.js格式
1. 在浏览器中加载和运行

**示例转换命令：**

```bash
# Step 1: PyTorch -> ONNX
python -c "
import torch
from deepmd.pt.model.model import get_model
model = get_model(...)
dummy_input = (coords, atype, box)
torch.onnx.export(model, dummy_input, 'model.onnx')
"

# Step 2: ONNX -> TensorFlow
onnx-tf convert -i model.onnx -o model_tf

# Step 3: TensorFlow -> TensorFlow.js
tensorflowjs_converter --input_format=tf_saved_model \
    model_tf web_model/
```

**优点：**

- ✅ 轻量级运行时（约1-2MB）
- ✅ WebGL加速支持
- ✅ 首次加载快

**缺点：**

- ⚠️ 需要离线转换模型
- ⚠️ 仅支持TensorFlow计算图
- ⚠️ 可能丢失部分功能

______________________________________________________________________

### 🔄 备选方案3：ONNX Runtime Web (Alternative: ONNX Runtime)

**状态：** 未实现

**实现步骤：**

1. 导出PyTorch模型为ONNX格式
1. 使用ONNX Runtime Web运行

**示例代码：**

```javascript
import * as ort from 'onnxruntime-web';

// Load model
const session = await ort.InferenceSession.create('model.onnx');

// Run inference
const feeds = {
    'coord': new ort.Tensor('float32', coords, [1, natoms, 3]),
    'atype': new ort.Tensor('int32', atypes, [natoms]),
    'box': new ort.Tensor('float32', box, [1, 9])
};

const results = await session.run(feeds);
const energy = results['energy'].data;
const force = results['force'].data;
const virial = results['virial'].data;
```

**优点：**

- ✅ 跨框架支持（PyTorch, TF, 等）
- ✅ WebGL/WebGPU加速
- ✅ 较小体积（约5-10MB）

**缺点：**

- ⚠️ 需要导出为ONNX
- ⚠️ 可能需要自定义算子

______________________________________________________________________

## 性能对比 (Performance Comparison)

| 方案          | 首次加载时间 | 推理速度 | 体积    | 开发难度 | 功能完整性 |
| ------------- | ------------ | -------- | ------- | -------- | ---------- |
| Pyodide       | ~30-60s      | 慢       | 60-80MB | 低       | ✅ 完整    |
| WASM          | ~5-10s       | 快       | 10-20MB | 高       | ✅ 完整    |
| TensorFlow.js | ~2-5s        | 中等     | 1-5MB   | 中       | ⚠️ 部分    |
| ONNX Runtime  | ~3-8s        | 快       | 5-10MB  | 中       | ⚠️ 部分    |

______________________________________________________________________

## 推荐使用场景 (Recommended Use Cases)

### 当前实现（Pyodide）适用于：

- ✅ 教育和演示
- ✅ 快速原型验证
- ✅ 小分子测试（< 50原子）
- ✅ 需要完整Python API

### 如需生产环境，建议：

- 🚀 **WASM方案** - 最佳性能，适合实时推理
- 🚀 **TensorFlow.js** - 快速加载，适合简单模型
- 🚀 **ONNX Runtime** - 平衡性能和兼容性

______________________________________________________________________

## 扩展建议 (Future Extensions)

### 短期改进：

1. ✅ 添加更多示例分子
1. ✅ 改进错误处理和用户反馈
1. ⚠️ 添加3D可视化（使用3Dmol.js或NGL Viewer）
1. ⚠️ 支持批量计算
1. ⚠️ 导出结果为JSON/CSV

### 长期改进：

1. ⚠️ 实现WASM版本以提升性能
1. ⚠️ 支持GPU加速（WebGL/WebGPU）
1. ⚠️ 添加轨迹分析功能
1. ⚠️ 支持更多模型格式
1. ⚠️ 集成分子动力学模拟

______________________________________________________________________

## 相关文件 (Related Files)

### Python推理实现：

- `deepmd/infer/deep_pot.py` - 高层Python API
- `deepmd/pt/infer/deep_eval.py` - PyTorch后端
- `deepmd/pt/train/wrapper.py` - 模型包装器

### C++推理实现：

- `source/api_cc/include/DeepPot.h` - C++ API头文件
- `source/api_cc/src/DeepPot.cc` - C++ API实现
- `source/api_c/include/c_api.h` - C API（WASM友好）

### 示例和测试：

- `examples/infer_water/infer_water.cpp` - C++推理示例
- `examples/water/se_e2_a/` - 训练配置示例
- `source/tests/pt/test_*.py` - PyTorch后端测试

______________________________________________________________________

## 技术细节 (Technical Details)

### 输入格式：

```python
coords: np.ndarray  # Shape: (nframes, natoms, 3), Unit: Å
cells: np.ndarray  # Shape: (nframes, 9) or None
atom_types: list[int]  # Length: natoms
```

### 输出格式：

```python
energy: np.ndarray  # Shape: (nframes, 1), Unit: eV
force: np.ndarray  # Shape: (nframes, natoms, 3), Unit: eV/Å
virial: np.ndarray  # Shape: (nframes, 9), Unit: eV
```

### 模型格式：

- `.pth` / `.pt` - PyTorch checkpoint或JIT模型
- `.pb` - TensorFlow SavedModel
- `.pdmodel` - Paddle模型

______________________________________________________________________

## 参考资源 (References)

### 官方文档：

- [DeePMD-kit Documentation](https://docs.deepmodeling.com/projects/deepmd/)
- [Pyodide Documentation](https://pyodide.org/)
- [TensorFlow.js Guide](https://www.tensorflow.org/js)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)

### 相关项目：

- [3Dmol.js](https://3dmol.csb.pitt.edu/) - 分子可视化
- [NGL Viewer](http://nglviewer.org/) - 高性能分子渲染
- [Emscripten](https://emscripten.org/) - C++到WASM编译器

______________________________________________________________________

## 贡献指南 (Contributing)

欢迎贡献改进！请参考：

1. Fork本仓库
1. 创建feature分支
1. 提交改进（代码、文档、示例）
1. 发起Pull Request

特别欢迎：

- WASM实现
- 性能优化
- 更多示例分子
- 3D可视化集成
- 更好的错误处理
