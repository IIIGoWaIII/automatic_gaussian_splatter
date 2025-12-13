const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const statusDiv = document.getElementById('pipelineStatus');
const consoleWindow = document.getElementById('consoleWindow');
const consoleOutput = document.getElementById('consoleOutput');
const videoSettings = document.getElementById('videoSettings');
const extractionValueInput = document.getElementById('extractionValue');
const valueLabel = document.getElementById('valueLabel');
const estimatedFramesDiv = document.getElementById('estimatedFrames');
const startUploadBtn = document.getElementById('startUploadBtn');
const colmapQuality = document.getElementById('colmapQuality');
const colmapDense = document.getElementById('colmapDense');
const brushSteps = document.getElementById('brushSteps');
const brushViewer = document.getElementById('brushViewer');
const brushShDegree = document.getElementById('brushShDegree');
const brushMaxSplats = document.getElementById('brushMaxSplats');
const brushMaxResolution = document.getElementById('brushMaxResolution');
const projectNameInput = document.getElementById('projectName');

// Resume Training Elements
const resumeProject = document.getElementById('resumeProject');
const resumeCheckpoint = document.getElementById('resumeCheckpoint');
const resumeTargetSteps = document.getElementById('resumeTargetSteps');
const resumeViewer = document.getElementById('resumeViewer');
const resumeTrainingBtn = document.getElementById('resumeTrainingBtn');
const refreshProjectsBtn = document.getElementById('refreshProjectsBtn');
const resumeForceScratch = document.getElementById('resumeForceScratch'); // New Checkbox

// Store available projects data
let availableProjects = [];

const steps = {
    stacking: document.getElementById('stepStacking'),
    tracking: document.getElementById('stepTracking'),
    training: document.getElementById('stepTraining')
};

// State
let selectedFiles = [];
let videoDuration = 0; // seconds

// Load default settings from backend so the UI mirrors current config
async function loadSettings() {
    try {
        const res = await fetch('/settings');
        if (!res.ok) throw new Error('Failed to load settings');
        const data = await res.json();
        if (data?.colmap) {
            if (data.colmap.quality) colmapQuality.value = data.colmap.quality;
            colmapDense.checked = Boolean(data.colmap.dense);
        }
        if (data?.brush) {
            if (data.brush.total_steps) brushSteps.value = data.brush.total_steps;
            brushViewer.checked = Boolean(data.brush.with_viewer);
            if (data.brush.sh_degree !== undefined) brushShDegree.value = data.brush.sh_degree;
            if (data.brush.max_splats) brushMaxSplats.value = data.brush.max_splats;
            if (data.brush.max_resolution) brushMaxResolution.value = data.brush.max_resolution;
        }
    } catch (err) {
        console.warn('Unable to fetch settings, using defaults', err);
    }
}
loadSettings();

// Load available projects for resume training
async function loadProjects() {
    try {
        const res = await fetch('/list-outputs');
        if (!res.ok) throw new Error('Failed to load projects');
        const data = await res.json();
        availableProjects = data.outputs || [];

        // Populate project dropdown
        resumeProject.innerHTML = '';
        if (availableProjects.length === 0) {
            resumeProject.innerHTML = '<option value="">-- No projects available --</option>';
            resumeCheckpoint.innerHTML = '<option value="">-- No checkpoints --</option>';
        } else {
            resumeProject.innerHTML = '<option value="">-- Select a project --</option>';
            availableProjects.forEach((proj, idx) => {
                const checkpointCount = proj.ply_checkpoints.length;
                const maxIter = checkpointCount > 0 ? proj.ply_checkpoints[checkpointCount - 1].iteration : 0;
                const opt = document.createElement('option');
                opt.value = idx;
                opt.textContent = `${proj.folder} (${checkpointCount} checkpoints, max: ${maxIter})`;
                resumeProject.appendChild(opt);
            });
        }
    } catch (err) {
        console.warn('Unable to fetch projects', err);
        resumeProject.innerHTML = '<option value="">-- Error loading projects --</option>';
    }
}
loadProjects();

// Handle Force Scratch Checkbox
resumeForceScratch.addEventListener('change', () => {
    if (resumeForceScratch.checked) {
        resumeCheckpoint.disabled = true;
        resumeCheckpoint.innerHTML = '<option value="">-- Ignored (Starting from Scratch) --</option>';
        resumeTrainingBtn.textContent = '▶ Start Training (Scratch)';
    } else {
        resumeCheckpoint.disabled = false;
        // Trigger change to repopulate if project is selected
        resumeProject.dispatchEvent(new Event('change'));
        resumeTrainingBtn.textContent = '▶ Resume Training';
    }
});

// Handle project selection - populate checkpoints
resumeProject.addEventListener('change', () => {
    const idx = parseInt(resumeProject.value, 10);

    // If scratch is on, we don't need to populate checkpoints
    if (resumeForceScratch.checked) {
        return;
    }

    resumeCheckpoint.innerHTML = '';

    if (isNaN(idx) || !availableProjects[idx]) {
        resumeCheckpoint.innerHTML = '<option value="">-- Select a project first --</option>';
        return;
    }

    const proj = availableProjects[idx];
    if (proj.ply_checkpoints.length === 0) {
        resumeCheckpoint.innerHTML = '<option value="">-- No checkpoints found --</option>';
        return;
    }

    proj.ply_checkpoints.forEach(cp => {
        const opt = document.createElement('option');
        opt.value = cp.iteration;
        opt.textContent = `Iteration ${cp.iteration.toLocaleString()} (${cp.filename})`;
        resumeCheckpoint.appendChild(opt);
    });

    // Auto-select highest checkpoint and set target steps
    const lastCheckpoint = proj.ply_checkpoints[proj.ply_checkpoints.length - 1];
    resumeCheckpoint.value = lastCheckpoint.iteration;
    resumeTargetSteps.value = lastCheckpoint.iteration + 5000;
});

// Refresh projects button
refreshProjectsBtn.addEventListener('click', loadProjects);

// Resume Training button
resumeTrainingBtn.addEventListener('click', async () => {
    const projIdx = parseInt(resumeProject.value, 10);
    let startIter = parseInt(resumeCheckpoint.value, 10);
    const targetSteps = parseInt(resumeTargetSteps.value, 10);
    const forceScratch = resumeForceScratch.checked;

    if (isNaN(projIdx) || !availableProjects[projIdx]) {
        alert('Please select a project first.');
        return;
    }

    if (!forceScratch && isNaN(startIter)) {
        alert('Please select a checkpoint to resume from.');
        return;
    }

    // If scratching, startIter is 0
    if (forceScratch) {
        startIter = 0;
    }

    if (isNaN(targetSteps) || targetSteps <= startIter) {
        alert(`Target steps (${targetSteps}) must be greater than start iteration (${startIter}).`);
        return;
    }

    const proj = availableProjects[projIdx];

    // Show console
    statusDiv.style.display = 'flex';
    consoleWindow.style.display = 'block';
    consoleOutput.innerHTML = '';

    // Mark training step as active
    updateStep('training');

    const formData = new FormData();
    formData.append('projectPath', proj.path);
    formData.append('startIter', startIter);
    formData.append('totalSteps', targetSteps);
    formData.append('forceScratch', forceScratch);

    const brushSettings = {
        with_viewer: resumeViewer.checked,
        sh_degree: parseInt(brushShDegree.value, 10),
        max_splats: parseInt(brushMaxSplats.value, 10),
        max_resolution: parseInt(brushMaxResolution.value, 10)
    };
    formData.append('brushSettings', JSON.stringify(brushSettings));

    try {
        const response = await fetch('/resume', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Request failed');

        const result = await response.json();
        console.log('Request started:', result);
    } catch (err) {
        console.error(err);
        alert('Error starting training. Check console.');
    }
});

// WebSocket setup
const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
const ws = new WebSocket(`${protocol}://${window.location.host}/ws`);

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    if (data.type === 'log') {
        const line = document.createElement('div');
        line.className = 'console-line';
        line.textContent = `> ${data.message}`;
        consoleOutput.appendChild(line);
        consoleWindow.scrollTop = consoleWindow.scrollHeight;

        // Simple heuristic to update steps based on logs
        if (data.message.includes('Step 1')) updateStep('stacking');
        if (data.message.includes('COLMAP') || data.message.includes('Step 2')) updateStep('tracking');
        if (data.message.includes('Brush') || data.message.includes('Step 3')) updateStep('training');
    }

    if (data.type === 'status') {
        if (data.status === 'completed') {
            const line = document.createElement('div');
            line.className = 'console-line';
            line.style.color = 'var(--success)';
            line.textContent = 'DONE! Output available in processing_output folder.';
            consoleOutput.appendChild(line);
            markAllCompleted();
        } else if (data.status === 'failed') {
            const line = document.createElement('div');
            line.className = 'console-line';
            line.style.color = 'var(--error)';
            line.textContent = 'FAILED! See logs above.';
            consoleOutput.appendChild(line);
        }
    }
};

function updateStep(activeStep) {
    // Reset all
    Object.values(steps).forEach(el => {
        el.classList.remove('active');
        el.classList.remove('completed');
    });

    if (activeStep === 'stacking') {
        steps.stacking.classList.add('active');
    } else if (activeStep === 'tracking') {
        steps.stacking.classList.add('completed');
        steps.tracking.classList.add('active');
    } else if (activeStep === 'training') {
        steps.stacking.classList.add('completed');
        steps.tracking.classList.add('completed');
        steps.training.classList.add('active');
    }
}

function markAllCompleted() {
    Object.values(steps).forEach(el => {
        el.classList.remove('active');
        el.classList.add('completed');
    });
}

// Video Settings Logic
document.querySelectorAll('input[name="extractionMode"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        if (e.target.value === 'fps') {
            valueLabel.textContent = 'Frames per Second';
            extractionValueInput.value = '2';
            extractionValueInput.step = '0.1';
        } else {
            valueLabel.textContent = 'Total Frame Count';
            extractionValueInput.value = '100';
            extractionValueInput.step = '1';
        }
        calculateEstimatedFrames();
    });
});

extractionValueInput.addEventListener('input', calculateEstimatedFrames);

function calculateEstimatedFrames() {
    const mode = document.querySelector('input[name="extractionMode"]:checked').value;
    const val = parseFloat(extractionValueInput.value);

    if (isNaN(val) || val <= 0) {
        estimatedFramesDiv.textContent = '-';
        return;
    }

    if (mode === 'count') {
        estimatedFramesDiv.textContent = `~ ${Math.round(val)} frames`;
    } else {
        if (!videoDuration) {
            estimatedFramesDiv.textContent = 'Loading video...';
        } else {
            const total = Math.round(videoDuration * val);
            estimatedFramesDiv.textContent = `~ ${total} frames`;
        }
    }
}

// File Handling
function handleFileSelection(files) {
    selectedFiles = Array.from(files);

    if (selectedFiles.length === 0) return;

    const isVideo = selectedFiles.length === 1 && selectedFiles[0].type.startsWith('video/');

    if (isVideo) {
        // Show Video Settings
        dropZone.style.display = 'none';
        videoSettings.style.display = 'block';

        // Get duration
        const video = document.createElement('video');
        video.preload = 'metadata';
        video.onloadedmetadata = function () {
            window.URL.revokeObjectURL(video.src);
            videoDuration = video.duration;
            calculateEstimatedFrames();
        }
        video.src = URL.createObjectURL(selectedFiles[0]);

    } else {
        // Images - Upload Immediately
        startUpload();
    }
}

startUploadBtn.addEventListener('click', startUpload);

async function startUpload() {
    if (!selectedFiles.length) {
        alert("Select images or a video first.");
        return;
    }

    // UI Updates
    dropZone.style.display = 'none';
    videoSettings.style.display = 'none';
    statusDiv.style.display = 'flex';
    consoleWindow.style.display = 'block';

    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });

    // Add video settings
    const mode = document.querySelector('input[name="extractionMode"]:checked').value;
    const val = extractionValueInput.value;
    formData.append('extractionMode', mode);
    formData.append('extractionValue', val);

    // Add project name if provided
    const projectName = projectNameInput.value.trim();
    if (projectName) {
        formData.append('projectName', projectName);
    }

    // Add COLMAP + Brush settings
    const colmapSettings = {
        quality: colmapQuality.value,
        dense: colmapDense.checked,
        sparse: 1
    };

    const brushSettings = {
        total_steps: parseInt(brushSteps.value, 10),
        with_viewer: brushViewer.checked,
        sh_degree: parseInt(brushShDegree.value, 10),
        max_splats: parseInt(brushMaxSplats.value, 10),
        max_resolution: parseInt(brushMaxResolution.value, 10)
    };

    formData.append('colmapSettings', JSON.stringify(colmapSettings));
    formData.append('brushSettings', JSON.stringify(brushSettings));

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error("Upload failed");

        const result = await response.json();
        console.log("Task started:", result.task_id);
    } catch (err) {
        console.error(err);
        alert("Error uploading file. Check console.");
    }
}

// Drag and Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFileSelection(e.dataTransfer.files);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleFileSelection(e.target.files);
    }
});
