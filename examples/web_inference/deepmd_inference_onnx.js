// SPDX-License-Identifier: LGPL-3.0-or-later
/**
 * DeePMD-kit Web Inference using ONNX Runtime Web
 * JavaScript interface for browser-based molecular dynamics inference
 */

let session = null;
let modelLoaded = false;
let modelInfo = {
  typeMap: [],
  rcut: 0,
};

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
 * Load ONNX model
 */
async function loadModel() {
  try {
    const modelUrl = document.getElementById('modelUrl').value.trim();

    if (!modelUrl) {
      updateStatus('请输入模型URL', 'error');
      setTimeout(hideStatus, 3000);
      return;
    }

    updateStatus('正在下载ONNX模型...', 'loading');
    document.getElementById('loadModelBtn').disabled = true;

    // Load the ONNX model
    session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    modelLoaded = true;
    document.getElementById('computeBtn').disabled = false;

    // Get model metadata
    const inputs = session.inputNames;
    const outputs = session.outputNames;

    updateStatus(
        `模型加载成功！<br>` +
            `输入: ${inputs.join(', ')}<br>` +
            `输出: ${outputs.join(', ')}`,
        'success');

    console.log('Model loaded:', {inputs, outputs});

  } catch (error) {
    console.error('Model loading error:', error);
    updateStatus(
        `模型加载失败: ${error.message}<br>
      <small>提示: 请先使用export_to_onnx.py将模型转换为ONNX格式</small>`,
        'error');
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
 * Compute properties using ONNX Runtime
 */
async function computeProperties() {
  if (!session || !modelLoaded) {
    updateStatus('请先加载模型', 'error');
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

    const natoms = coords.length;
    const nframes = 1;

    // Prepare coordinate tensor (nframes, natoms, 3)
    const coordData = new Float64Array(nframes * natoms * 3);
    for (let i = 0; i < natoms; i++) {
      coordData[i * 3 + 0] = coords[i][0];
      coordData[i * 3 + 1] = coords[i][1];
      coordData[i * 3 + 2] = coords[i][2];
    }
    const coordTensor =
        new ort.Tensor('float64', coordData, [nframes, natoms, 3]);

    // Prepare atom type tensor (natoms,)
    const atypeData = new Int32Array(atomTypes);
    const atypeTensor = new ort.Tensor('int32', atypeData, [natoms]);

    // Prepare box tensor (nframes, 3, 3)
    const boxData = new Float64Array(nframes * 9);
    if (cell && cell.length === 3) {
      for (let i = 0; i < 3; i++) {
        boxData[i * 3 + 0] = cell[i][0];
        boxData[i * 3 + 1] = cell[i][1];
        boxData[i * 3 + 2] = cell[i][2];
      }
    }
    const boxTensor = new ort.Tensor('float64', boxData, [nframes, 3, 3]);

    console.log('Running inference with:', {
      coord: coordTensor.dims,
      atype: atypeTensor.dims,
      box: boxTensor.dims,
    });

    // Run inference
    const feeds = {
      coord: coordTensor,
      atype: atypeTensor,
      box: boxTensor,
    };

    const results = await session.run(feeds);

    // Extract results
    const energy = results.energy.data;
    const force = results.force.data;
    const virial = results.virial.data;

    console.log('Inference results:', {
      energy: energy.length,
      force: force.length,
      virial: virial.length,
    });

    // Reshape force data (nframes, natoms, 3)
    const forceArray = [];
    for (let i = 0; i < natoms; i++) {
      forceArray.push([force[i * 3 + 0], force[i * 3 + 1], force[i * 3 + 2]]);
    }

    // Display results
    displayResults({
      success: true,
      energy: [[energy[0]]],
      force: [forceArray],
      virial: [Array.from(virial)],
      natoms: natoms,
      nframes: 1,
    });

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

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
  console.log('DeePMD-kit Web Inference (ONNX Runtime) loaded');
  updateStatus(
      '欢迎使用DeePMD-kit浏览器推理！请先加载ONNX模型。<br>' +
          '<small>使用export_to_onnx.py转换模型</small>',
      'loading');
});
