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
const colmapRemoveDuplicates = document.getElementById('colmapRemoveDuplicates');
const brushSteps = document.getElementById('brushSteps');
const brushViewer = document.getElementById('brushViewer');
const brushShutdown = document.getElementById('brushShutdown');
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
const resumeShutdown = document.getElementById('resumeShutdown');

// Update Modal Elements
const updateModal = document.getElementById('updateModal');
const updateList = document.getElementById('updateList');
const updateCancelBtn = document.getElementById('updateCancelBtn');
const updateConfirmBtn = document.getElementById('updateConfirmBtn');

// Store available updates
let pendingUpdates = [];

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
            colmapRemoveDuplicates.checked = Boolean(data.colmap.remove_duplicates);
        }
        if (data?.brush) {
            if (data.brush.total_steps) brushSteps.value = data.brush.total_steps;
            brushViewer.checked = Boolean(data.brush.with_viewer);
            brushShutdown.checked = Boolean(data.brush.shutdown_after_training);
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

// Check for updates on page load
async function checkForUpdates() {
    try {
        const res = await fetch('/check-updates');
        if (!res.ok) throw new Error('Failed to check updates');
        const data = await res.json();

        if (data.updates_available && data.updates.length > 0) {
            pendingUpdates = data.updates;
            showUpdateModal(data.updates);
        }
    } catch (err) {
        console.warn('Unable to check for updates', err);
    }
}

function showUpdateModal(updates) {
    // Clear previous content
    updateList.innerHTML = '';

    // Separate updates into categories
    const toolUpdates = updates.filter(u => u.key === 'colmap' || u.key === 'brush');
    const appUpdates = updates.filter(u => u.key === 'app');

    // Add section header for tools if there are tool updates
    if (toolUpdates.length > 0) {
        const toolHeader = document.createElement('div');
        toolHeader.className = 'update-section-header';
        toolHeader.textContent = 'Tools (COLMAP & Brush)';
        updateList.appendChild(toolHeader);

        toolUpdates.forEach(update => {
            const item = document.createElement('label');
            item.className = 'update-item update-item-selectable';
            item.innerHTML = `
                <input type="checkbox" class="update-checkbox" data-key="${update.key}" checked>
                <span class="update-item-name">${update.name}</span>
                <span class="update-item-versions">
                    <span class="current">${update.current}</span>
                    <span class="arrow">→</span>
                    <span class="latest">${update.latest}</span>
                </span>
            `;
            updateList.appendChild(item);
        });
    }

    // Add section header for app if there are app updates
    if (appUpdates.length > 0) {
        const appHeader = document.createElement('div');
        appHeader.className = 'update-section-header';
        appHeader.textContent = 'WebUI Application';
        updateList.appendChild(appHeader);

        appUpdates.forEach(update => {
            const item = document.createElement('label');
            item.className = 'update-item update-item-selectable';
            item.innerHTML = `
                <input type="checkbox" class="update-checkbox" data-key="${update.key}" checked>
                <span class="update-item-name">${update.name}</span>
                <span class="update-item-versions">
                    <span class="current">${update.current}</span>
                    <span class="arrow">→</span>
                    <span class="latest">${update.latest}</span>
                </span>
            `;
            updateList.appendChild(item);
        });
    }

    // Show modal
    updateModal.style.display = 'flex';
}

function hideUpdateModal() {
    updateModal.style.display = 'none';
    // Reset button states
    updateConfirmBtn.disabled = false;
    updateConfirmBtn.textContent = 'Update Selected';
    updateCancelBtn.style.display = 'inline-block';
}

function getSelectedUpdates() {
    const checkboxes = updateList.querySelectorAll('.update-checkbox:checked');
    const selectedKeys = Array.from(checkboxes).map(cb => cb.dataset.key);
    return pendingUpdates.filter(u => selectedKeys.includes(u.key));
}

// Update modal button handlers
updateCancelBtn.addEventListener('click', hideUpdateModal);

updateConfirmBtn.addEventListener('click', async () => {
    const selectedUpdates = getSelectedUpdates();
    if (selectedUpdates.length === 0) {
        alert('Please select at least one component to update.');
        return;
    }

    // Disable buttons and show progress
    updateConfirmBtn.disabled = true;
    updateConfirmBtn.textContent = 'Updating...';
    updateCancelBtn.style.display = 'none';

    // Replace update list with progress indicator
    updateList.innerHTML = `
        <div class="update-progress">
            <span class="spinner"></span>
            Installing ${selectedUpdates.length} update(s)... Check console for progress.
        </div>
    `;

    // Show console for progress logs
    statusDiv.style.display = 'flex';
    consoleWindow.style.display = 'block';
    consoleOutput.innerHTML = '';

    try {
        const formData = new FormData();
        formData.append('updates', JSON.stringify(selectedUpdates));

        const response = await fetch('/install-updates', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Update request failed');

        const result = await response.json();
        console.log('Update started:', result);

        // Modal will be closed when we receive completion status via WebSocket
    } catch (err) {
        console.error('Update failed:', err);
        updateList.innerHTML = `
            <div class="update-progress" style="color: var(--error);">
                Update failed. Please try again later.
            </div>
        `;
        updateConfirmBtn.textContent = 'Update Selected';
        updateConfirmBtn.disabled = false;
        updateCancelBtn.style.display = 'inline-block';
    }
});

// Check for updates after a short delay to let page load
setTimeout(checkForUpdates, 1000);

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
        shutdown_after_training: resumeShutdown.checked,
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
        // Output to multi-image console
        const line = document.createElement('div');
        line.className = 'console-line';
        line.textContent = `> ${data.message}`;
        consoleOutput.appendChild(line);
        consoleWindow.scrollTop = consoleWindow.scrollHeight;

        // Also output to single-image console if visible
        if (singleConsoleWindow && singleConsoleWindow.style.display !== 'none') {
            const singleLine = document.createElement('div');
            singleLine.className = 'console-line';
            singleLine.textContent = `> ${data.message}`;
            singleConsoleOutput.appendChild(singleLine);
            singleConsoleWindow.scrollTop = singleConsoleWindow.scrollHeight;
        }

        // Simple heuristic to update steps based on logs
        if (data.message.includes('Step 1')) updateStep('stacking');
        if (data.message.includes('COLMAP') || data.message.includes('Step 2')) updateStep('tracking');
        if (data.message.includes('Brush') || data.message.includes('Step 3')) updateStep('training');

        // Single workflow step update
        if (data.message.includes('SHARP')) {
            singleStepProcess.classList.add('active');
        }
    }

    if (data.type === 'status') {
        // Handle update completion
        if (data.task_id === 'update') {
            if (data.status === 'completed' || data.status === 'partial') {
                hideUpdateModal();
                pendingUpdates = [];
                const line = document.createElement('div');
                line.className = 'console-line';
                line.style.color = data.status === 'completed' ? 'var(--success)' : 'var(--error)';
                line.textContent = data.status === 'completed'
                    ? 'Updates installed successfully! You may need to restart the app.'
                    : 'Some updates failed. Check logs above.';
                consoleOutput.appendChild(line);
            }
            return;
        }

        if (data.status === 'completed') {
            // Multi-image workflow
            const line = document.createElement('div');
            line.className = 'console-line';
            line.style.color = 'var(--success)';
            line.textContent = 'DONE! Output available in processing_output folder.';
            consoleOutput.appendChild(line);
            markAllCompleted();

            // Single image workflow
            if (singleConsoleWindow && singleConsoleWindow.style.display !== 'none') {
                const singleLine = document.createElement('div');
                singleLine.className = 'console-line';
                singleLine.style.color = 'var(--success)';
                singleLine.textContent = 'DONE! 3D Gaussians saved to processing_output folder.';
                singleConsoleOutput.appendChild(singleLine);
                singleStepProcess.classList.remove('active');
                singleStepProcess.classList.add('completed');
            }
        } else if (data.status === 'failed') {
            const line = document.createElement('div');
            line.className = 'console-line';
            line.style.color = 'var(--error)';
            line.textContent = 'FAILED! See logs above.';
            consoleOutput.appendChild(line);

            // Single image workflow
            if (singleConsoleWindow && singleConsoleWindow.style.display !== 'none') {
                const singleLine = document.createElement('div');
                singleLine.className = 'console-line';
                singleLine.style.color = 'var(--error)';
                singleLine.textContent = 'FAILED! See logs above.';
                singleConsoleOutput.appendChild(singleLine);
            }
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
        remove_duplicates: colmapRemoveDuplicates.checked,
        sparse: 1
    };

    const brushSettings = {
        total_steps: parseInt(brushSteps.value, 10),
        with_viewer: brushViewer.checked,
        shutdown_after_training: brushShutdown.checked,
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

// ==========================================
// SINGLE IMAGE WORKFLOW (Sharp)
// ==========================================

const singleDropZone = document.getElementById('singleDropZone');
const singleFileInput = document.getElementById('singleFileInput');
const singleProjectName = document.getElementById('singleProjectName');
const sharpDevice = document.getElementById('sharpDevice');
const singlePipelineStatus = document.getElementById('singlePipelineStatus');
const singleConsoleWindow = document.getElementById('singleConsoleWindow');
const singleConsoleOutput = document.getElementById('singleConsoleOutput');
const singleStepProcess = document.getElementById('singleStepProcess');

// Handle single image selection
function handleSingleImageSelection(files) {
    if (files.length === 0) return;

    const file = files[0];
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file (JPG, PNG, or WEBP)');
        return;
    }

    startSingleUpload(file);
}

async function startSingleUpload(file) {
    // Show console
    singleDropZone.style.display = 'none';
    singlePipelineStatus.style.display = 'flex';
    singleConsoleWindow.style.display = 'block';
    singleConsoleOutput.innerHTML = '';
    singleStepProcess.classList.add('active');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('device', sharpDevice.value);
    formData.append('render', 'false');

    const projectName = singleProjectName.value.trim();
    if (projectName) {
        formData.append('projectName', projectName);
    }

    try {
        const response = await fetch('/upload-single', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Upload failed');

        const result = await response.json();
        console.log('Sharp task started:', result.task_id);
    } catch (err) {
        console.error(err);
        alert('Error uploading image. Check console.');
        resetSingleWorkflow();
    }
}

function resetSingleWorkflow() {
    singleDropZone.style.display = 'block';
    singlePipelineStatus.style.display = 'none';
    singleConsoleWindow.style.display = 'none';
    singleStepProcess.classList.remove('active', 'completed');
}

// Drag and Drop for Single Image
singleDropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    singleDropZone.classList.add('dragover');
});

singleDropZone.addEventListener('dragleave', () => {
    singleDropZone.classList.remove('dragover');
});

singleDropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    singleDropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleSingleImageSelection(e.dataTransfer.files);
    }
});

singleFileInput.addEventListener('change', (e) => {
    if (e.target.files.length) {
        handleSingleImageSelection(e.target.files);
    }
});
