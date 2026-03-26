// SPDX-License-Identifier: LGPL-3.0-or-later
/**
 * DeePMD-kit Web Inference
 * JavaScript interface for browser-based molecular dynamics inference
 */

let pyodide = null;
let modelLoaded = false;

/**
 * Update status display
 */
function updateStatus(message, type = 'loading') {
  const statusDiv = document.getElementById('status');
  statusDiv.style.display = 'block';
  statusDiv.className = `status ${type}`;

  if (type === 'loading') {
    statusDiv.innerHTML = `<span class="spinner"></span> ${message}`;
  } else {
    statusDiv.innerHTML = message;
  }
}

/**
 * Hide status display
 */
function hideStatus() {
  document.getElementById('status').style.display = 'none';
}

/**
 * Initialize Pyodide environment
 */
async function initializePyodide() {
  try {
    updateStatus(
        '正在初始化Pyodide环境（首次加载需要下载~60MB，请耐心等待）...',
        'loading');

    // Disable button during initialization
    document.getElementById('initPyodideBtn').disabled = true;

    // Load Pyodide
    pyodide = await loadPyodide(
        {indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/'});

    updateStatus('正在安装NumPy...', 'loading');
    await pyodide.loadPackage(['numpy', 'micropip']);

    updateStatus('正在配置DeePMD-kit安装...', 'loading');

    // Install deepmd-kit via micropip
    const micropip = pyodide.pyimport('micropip');

    // Try to install deepmd-kit and torch
    try {
      updateStatus('正在安装PyTorch (CPU版本)...', 'loading');
      // Install torch CPU version
      await micropip.install('torch==2.1.0', {keep_going: true});

      updateStatus('正在安装DeePMD-kit...', 'loading');
      // Install deepmd-kit
      await micropip.install('deepmd-kit', {keep_going: true});

      updateStatus('Pyodide环境初始化成功！现在可以加载模型了。', 'success');

    } catch (installError) {
      console.error('Installation error:', installError);
      updateStatus('警告：部分包安装失败，尝试使用基础功能...', 'loading');

      // Load the inference script anyway
      await pyodide.runPythonAsync(`
import sys
print("Python version:", sys.version)
print("Available packages:", sys.modules.keys())
            `);

      updateStatus(
          '环境部分初始化完成。注意：完整功能可能需要本地安装DeePMD-kit。',
          'success');
    }

    // Enable model loading button
    document.getElementById('loadModelBtn').disabled = false;

    setTimeout(hideStatus, 3000);

  } catch (error) {
    console.error('Pyodide initialization error:', error);
    updateStatus(`初始化失败: ${error.message}`, 'error');
    document.getElementById('initPyodideBtn').disabled = false;
  }
}

/**
 * Load model
 */
async function loadModel() {
  if (!pyodide) {
    updateStatus('请先初始化Pyodide环境', 'error');
    setTimeout(hideStatus, 3000);
    return;
  }

  try {
    const modelUrl = document.getElementById('modelUrl').value.trim();

    if (!modelUrl) {
      updateStatus('请输入模型URL', 'error');
      setTimeout(hideStatus, 3000);
      return;
    }

    updateStatus('正在下载模型文件...', 'loading');
    document.getElementById('loadModelBtn').disabled = true;

    // Download model file
    const response = await fetch(modelUrl);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const modelData = await response.arrayBuffer();
    const modelBytes = new Uint8Array(modelData);

    updateStatus('正在加载模型到DeePMD-kit...', 'loading');

    // Write model to virtual filesystem
    pyodide.FS.writeFile('/model.pth', modelBytes);

    // Load model using deepmd
    await pyodide.runPythonAsync(`
import sys
import traceback

try:
    from deepmd.infer import DeepPot
    import numpy as np

    # Load the model
    dp = DeepPot('/model.pth')

    # Get model info
    type_map = dp.get_type_map()
    rcut = dp.get_rcut()
    ntypes = dp.get_ntypes()

    model_info = {
        'type_map': type_map,
        'rcut': rcut,
        'ntypes': ntypes,
        'loaded': True
    }

    print(f"Model loaded successfully!")
    print(f"Type map: {type_map}")
    print(f"Cutoff radius: {rcut} Å")
    print(f"Number of types: {ntypes}")

except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()
    model_info = {'loaded': False, 'error': str(e)}
`);

    const modelInfo = pyodide.globals.get('model_info').toJs();

    if (modelInfo.loaded) {
      modelLoaded = true;
      document.getElementById('computeBtn').disabled = false;

      const typeMapStr = Array.from(modelInfo.type_map || []).join(', ');
      updateStatus(
          `模型加载成功！<br>` +
              `类型映射: ${typeMapStr}<br>` +
              `截断半径: ${modelInfo.rcut} Å<br>` +
              `类型数量: ${modelInfo.ntypes}`,
          'success');
    } else {
      throw new Error(modelInfo.error || '未知错误');
    }

  } catch (error) {
    console.error('Model loading error:', error);
    updateStatus(`模型加载失败: ${error.message}`, 'error');
    document.getElementById('loadModelBtn').disabled = false;
  }
}

/**
 * Parse coordinates from textarea
 */
function parseCoordinates(text) {
  const lines = text.trim().split('\n');
  const coords = [];

  for (const line of lines) {
    const values = line.trim().split(/\s+/).map(parseFloat);
    if (values.length === 3 && values.every(v => !isNaN(v))) {
      coords.push(values);
    }
  }

  return coords;
}

/**
 * Parse atom types from input
 */
function parseAtomTypes(text) {
  return text.trim().split(/\s+/).map(v => parseInt(v)).filter(v => !isNaN(v));
}

/**
 * Compute properties
 */
async function computeProperties() {
  if (!pyodide || !modelLoaded) {
    updateStatus('请先初始化环境并加载模型', 'error');
    setTimeout(hideStatus, 3000);
    return;
  }

  try {
    updateStatus('正在计算...', 'loading');
    document.getElementById('computeBtn').disabled = true;

    // Parse input
    const atomTypes =
        parseAtomTypes(document.getElementById('atomTypes').value);
    const coords = parseCoordinates(document.getElementById('coords').value);
    const cellText = document.getElementById('cell').value.trim();
    const cell = cellText ? parseCoordinates(cellText) : null;

    // Validation
    if (atomTypes.length === 0) {
      throw new Error('请输入有效的原子类型');
    }

    if (coords.length === 0) {
      throw new Error('请输入有效的坐标');
    }

    if (coords.length !== atomTypes.length) {
      throw new Error(`坐标数量(${coords.length})与原子类型数量(${
          atomTypes.length})不匹配`);
    }

    if (cell && cell.length !== 3) {
      throw new Error('晶胞参数应该包含3行（3x3矩阵）');
    }

    // Convert to Python
    pyodide.globals.set('atom_types_js', atomTypes);
    pyodide.globals.set('coords_js', coords.flat());
    if (cell) {
      pyodide.globals.set('cell_js', cell.flat());
    }

    // Run inference
    const result = await pyodide.runPythonAsync(`
import numpy as np
import traceback

try:
    # Prepare input
    natoms = len(atom_types_js)
    coords = np.array(coords_js).reshape(1, natoms, 3)
    atom_types = list(atom_types_js)

    # Cell (if provided)
    if 'cell_js' in dir():
        cell = np.array(cell_js).reshape(1, 3, 3)
        # Convert to DeePMD format (1, 9)
        cell = cell.reshape(1, 9)
    else:
        cell = None

    # Run inference
    energy, force, virial, atom_energy, atom_virial = dp.eval(
        coords=coords,
        cells=cell,
        atom_types=atom_types,
        atomic=False
    )

    result = {
        'success': True,
        'energy': energy.tolist(),
        'force': force.tolist(),
        'virial': virial.tolist(),
        'natoms': natoms,
        'nframes': 1
    }

except Exception as e:
    print(f"Inference error: {e}")
    traceback.print_exc()
    result = {
        'success': False,
        'error': str(e)
    }

result
`);

    const resultObj = result.toJs();

    if (!resultObj.success) {
      throw new Error(resultObj.error || '计算失败');
    }

    // Display results
    displayResults(resultObj);

    updateStatus('计算完成！', 'success');
    setTimeout(hideStatus, 2000);

  } catch (error) {
    console.error('Computation error:', error);
    updateStatus(`计算失败: ${error.message}`, 'error');
  } finally {
    document.getElementById('computeBtn').disabled = false;
  }
}

/**
 * Display results
 */
function displayResults(result) {
  const resultsSection = document.getElementById('resultsSection');
  resultsSection.style.display = 'block';

  // Energy
  const energy = result.energy[0][0];
  document.getElementById('energyResult').textContent =
      `总能量: ${energy.toFixed(6)} eV\n` +
      `平均能量/原子: ${(energy / result.natoms).toFixed(6)} eV/atom`;

  // Force
  const forces = result.force[0];
  let forceText = 'Force (eV/Å):\n';
  forces.forEach((f, i) => {
    forceText += `原子 ${i}: [${f[0].toFixed(6)}, ${f[1].toFixed(6)}, ${
        f[2].toFixed(6)}]\n`;
  });
  document.getElementById('forceResult').textContent = forceText;

  // Virial
  const virial = result.virial[0];
  document.getElementById('virialResult').textContent =
      `维里张量 (eV):\n` +
      `[${virial[0].toFixed(6)}, ${virial[1].toFixed(6)}, ${
          virial[2].toFixed(6)}]\n` +
      `[${virial[3].toFixed(6)}, ${virial[4].toFixed(6)}, ${
          virial[5].toFixed(6)}]\n` +
      `[${virial[6].toFixed(6)}, ${virial[7].toFixed(6)}, ${
          virial[8].toFixed(6)}]`;

  // Info
  document.getElementById('computeInfo').textContent =
      `原子数: ${result.natoms}\n` +
      `帧数: ${result.nframes}\n` +
      `计算时间: ${new Date().toLocaleString()}`;

  // Scroll to results
  resultsSection.scrollIntoView({behavior: 'smooth', block: 'nearest'});
}

/**
 * Load example molecules
 */
function loadExample(type) {
  const examples = {
    water: {
      atomTypes: '0 1 1',
      coords: `0.0 0.0 0.0
0.0 0.0 0.96
0.0 0.93 -0.24`,
      cell: ''
    },
    water_box: {
      atomTypes: '0 1 1',
      coords: `5.0 5.0 5.0
5.0 5.0 5.96
5.0 5.93 4.76`,
      cell: `10.0 0.0 0.0
0.0 10.0 0.0
0.0 0.0 10.0`
    },
    ethanol: {
      atomTypes: '0 0 1 1 1 1 1 1 2',
      coords: `0.0 0.0 0.0
1.54 0.0 0.0
-0.37 1.03 0.0
-0.37 -0.52 0.89
-0.37 -0.52 -0.89
1.91 0.52 0.89
1.91 0.52 -0.89
1.91 -1.03 0.0
2.59 0.0 0.0`,
      cell: ''
    }
  };

  if (examples[type]) {
    document.getElementById('atomTypes').value = examples[type].atomTypes;
    document.getElementById('coords').value = examples[type].coords;
    document.getElementById('cell').value = examples[type].cell;

    updateStatus(
        `已加载${
            type === 'water'         ? '水分子' :
                type === 'water_box' ? '水盒子' :
                                       '乙醇分子'}示例`,
        'success');
    setTimeout(hideStatus, 2000);
  }
}

// Auto-initialize on page load
window.addEventListener('DOMContentLoaded', () => {
  console.log('DeePMD-kit Web Inference loaded');
  updateStatus(
      '欢迎使用DeePMD-kit浏览器推理！点击"初始化Pyodide环境"开始。', 'loading');
});
