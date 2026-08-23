// ==========================================
// Element refs
// ==========================================
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const dropSummary = document.getElementById('dropSummary');
const dropSummaryIcon = document.getElementById('dropSummaryIcon');
const dropSummaryText = document.getElementById('dropSummaryText');
const dropSummarySub = document.getElementById('dropSummarySub');
const changeFilesBtn = document.getElementById('changeFilesBtn');
const uploadCard = document.getElementById('uploadCard');

const multiWorkflow = document.getElementById('workflow-multiple');
const singleWorkflow = document.getElementById('workflow-single');

// LiDAR workflow elements
const lidarWorkflow = document.getElementById('workflow-lidar');
const lidarZipName = document.getElementById('lidarZipName');
const lidarProjectName = document.getElementById('lidarProjectName');
const lidarChangeBtn = document.getElementById('lidarChangeBtn');
const startLidarBtn = document.getElementById('startLidarBtn');
const lidarRefinePoses = document.getElementById('lidarRefinePoses');
const lidarTrainerType = document.getElementById('lidarTrainerType');
const lidarBrushSteps = document.getElementById('lidarBrushSteps');
const lidarMaxSplats = document.getElementById('lidarMaxSplats');
const lidarViewer = document.getElementById('lidarViewer');
const lidarShutdown = document.getElementById('lidarShutdown');
const lidarAutoSettingsChips = document.getElementById('lidarAutoSettingsChips');

// LiDAR training presets per scene type
const LIDAR_SCENARIOS = {
    object: { label: 'Single Object', steps: 30000, maxSplats: 3000000 },
    indoor: { label: 'Indoor Room', steps: 50000, maxSplats: 5000000 },
    outdoor: { label: 'Outdoor Scene', steps: 70000, maxSplats: 8000000 }
};

function applyLidarScenario(key) {
    const preset = LIDAR_SCENARIOS[key];
    if (!preset) return;
    lidarBrushSteps.value = preset.steps;
    lidarMaxSplats.value = preset.maxSplats;
    lidarAutoSettingsChips.innerHTML = `
        <span class="settings-chip">🏷️ Scene: ${preset.label}</span>
        <span class="settings-chip">⏱️ Steps: ${preset.steps.toLocaleString()}</span>
        <span class="settings-chip">🌀 Max splats: ${(preset.maxSplats / 1e6)}M</span>
    `;
}

document.querySelectorAll('input[name="lidarScenario"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        if (e.target.checked) applyLidarScenario(e.target.value);
    });
});

const statusDiv = document.getElementById('pipelineStatus');
const consoleWindow = document.getElementById('consoleWindow');
const consoleOutput = document.getElementById('consoleOutput');

const startUploadBtn = document.getElementById('startUploadBtn');
const autoSettingsChips = document.getElementById('autoSettingsChips');

const colmapQuality = document.getElementById('colmapQuality');
const colmapDense = document.getElementById('colmapDense');
const colmapRemoveDuplicates = document.getElementById('colmapRemoveDuplicates');
const colmapEngine = document.getElementById('colmapEngine');
const colmapMatcher = document.getElementById('colmapMatcher');
const trainerType = document.getElementById('trainerType');
const brushSteps = document.getElementById('brushSteps');
const brushViewer = document.getElementById('brushViewer');
const brushShutdown = document.getElementById('brushShutdown');
const brushShDegree = document.getElementById('brushShDegree');
const brushMaxSplats = document.getElementById('brushMaxSplats');
const brushMaxResolution = document.getElementById('brushMaxResolution');
const projectNameInput = document.getElementById('projectName');

// Single workflow elements
const singlePreviewImg = document.getElementById('singlePreviewImg');
const singlePreviewPlaceholder = document.getElementById('singlePreviewPlaceholder');
const singleProjectName = document.getElementById('singleProjectName');
const sharpDevice = document.getElementById('sharpDevice');
const singlePipelineStatus = document.getElementById('singlePipelineStatus');
const singleConsoleWindow = document.getElementById('singleConsoleWindow');
const singleConsoleOutput = document.getElementById('singleConsoleOutput');
const singleStepProcess = document.getElementById('singleStepProcess');
const startSingleBtn = document.getElementById('startSingleBtn');
const singleChangeBtn = document.getElementById('singleChangeBtn');
const singleFileInput = document.getElementById('singleFileInput');

// Resume Training Elements
const resumeProject = document.getElementById('resumeProject');
const resumeCheckpoint = document.getElementById('resumeCheckpoint');
const resumeTrainingBtn = document.getElementById('resumeTrainingBtn');
const refreshProjectsBtn = document.getElementById('refreshProjectsBtn');
const resumeForceScratch = document.getElementById('resumeForceScratch');

// Preview Elements
const splatList = document.getElementById('splatList');
const refreshSplatsBtn = document.getElementById('refreshSplatsBtn');

// Library toggle Elements
const libraryToggleBtn = document.getElementById('libraryToggleBtn');
const libraryCard = document.querySelector('.library');

// Pipeline progress Elements
const pipelineProgress = document.getElementById('pipelineProgress');
const progressNote = document.getElementById('progressNote');
const progressFill = document.getElementById('progressFill');
const progressPercent = document.getElementById('progressPercent');
const progressEta = document.getElementById('progressEta');
const advancedInfo = document.getElementById('advancedInfo');

// Update Modal Elements
const updateModal = document.getElementById('updateModal');
const updateList = document.getElementById('updateList');
const updateCancelBtn = document.getElementById('updateCancelBtn');
const updateConfirmBtn = document.getElementById('updateConfirmBtn');

// Too many images modal
const imageCountModal = document.getElementById('imageCountModal');
const imageCountMessage = document.getElementById('imageCountMessage');
const imageCountEstimate = document.getElementById('imageCountEstimate');
const useAllImagesBtn = document.getElementById('useAllImagesBtn');
const useFewerImagesBtn = document.getElementById('useFewerImagesBtn');

// Store available updates
let pendingUpdates = [];

// Store available projects data
let availableProjects = [];

const steps = {
    stacking: document.getElementById('stepStacking'),
    tracking: document.getElementById('stepTracking'),
    training: document.getElementById('stepTraining')
};

// ==========================================
// State
// ==========================================
let selectedFiles = [];
let lidarZipFile = null; // selected .zip LiDAR capture
let videoDuration = 0; // seconds
let autoExtractionFps = 2; // computed from video duration to hit 300-400 frames
let pendingUpload = null; // { files, sampled }
let singleImageFile = null;

const TARGET_FRAMES = 350; // 300-400 sweet spot
const MIN_FPS = 0.5;
const MAX_FPS = 6;
const MAX_IMAGE_COUNT = 400;

// ==========================================
// Scenario presets (based on community best practices)
// ==========================================
const SCENARIOS = {
    object: {
        label: 'Single Object',
        matcher: 'exhaustive',
        quality: 'high',
        steps: 30000,
        shDegree: 3,
        maxSplats: 3000000
    },
    indoor: {
        label: 'Indoor Room',
        matcher: 'exhaustive',
        quality: 'high',
        steps: 50000,
        shDegree: 3,
        maxSplats: 5000000
    },
    outdoor: {
        label: 'Outdoor Scene',
        matcher: 'exhaustive',
        quality: 'high',
        steps: 70000,
        shDegree: 3,
        maxSplats: 8000000
    },
    video: {
        label: 'Video Walkaround',
        matcher: 'sequential',
        quality: 'high',
        steps: 40000,
        shDegree: 3,
        maxSplats: 5000000
    }
};

function applyScenario(key) {
    if (key === 'custom') {
        autoSettingsChips.innerHTML = '<span class="settings-chip">⚙️ Using your manual Advanced Settings</span>';
        return;
    }
    const preset = SCENARIOS[key];
    if (!preset) return;

    colmapMatcher.value = preset.matcher;
    colmapQuality.value = preset.quality;
    brushSteps.value = preset.steps;
    brushShDegree.value = preset.shDegree;
    brushMaxSplats.value = preset.maxSplats;

    const matcherLabel = { exhaustive: 'Exhaustive', sequential: 'Sequential', auto: 'Auto' }[preset.matcher];
    autoSettingsChips.innerHTML = `
        <span class="settings-chip">🧭 Matcher: ${matcherLabel}</span>
        <span class="settings-chip">🎯 Quality: ${preset.quality[0].toUpperCase() + preset.quality.slice(1)}</span>
        <span class="settings-chip">⏱️ Steps: ${preset.steps.toLocaleString()}</span>
        <span class="settings-chip">💡 SH degree: ${preset.shDegree}</span>
        <span class="settings-chip">🌀 Max splats: ${(preset.maxSplats / 1e6)}M</span>
    `;
}

document.querySelectorAll('input[name="scenario"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        if (e.target.checked) applyScenario(e.target.value);
    });
});

// ==========================================
// Load default settings from backend
// ==========================================
async function loadSettings() {
    try {
        const res = await fetch('/settings');
        if (!res.ok) throw new Error('Failed to load settings');
        const data = await res.json();
        if (data?.colmap) {
            if (data.colmap.engine) colmapEngine.value = data.colmap.engine;
            if (data.colmap.matcher) colmapMatcher.value = data.colmap.matcher;
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

// ==========================================
// Load available projects for resume training
// ==========================================
async function loadProjects() {
    try {
        const res = await fetch('/list-outputs');
        if (!res.ok) throw new Error('Failed to load projects');
        const data = await res.json();
        availableProjects = data.outputs || [];

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

// ==========================================
// Preview splats (open finished splats in LichtFeld-Studio)
// ==========================================
async function loadSplats() {
    splatList.innerHTML = '<p class="settings-hint" style="margin-top: 1rem;">Loading...</p>';
    try {
        const res = await fetch('/list-splats');
        if (!res.ok) throw new Error('Failed to load splats');
        const data = await res.json();
        const splats = data.splats || [];

        if (splats.length === 0) {
            splatList.innerHTML = '<div class="splat-empty">No finished splats found yet.<br>Complete a pipeline and the output will show up here.</div>';
            return;
        }

        splatList.innerHTML = '';
        splats.forEach(splat => {
            const item = document.createElement('div');
            item.className = 'splat-item';
            item.dataset.path = splat.path;
            const ext = splat.filename.split('.').pop().toUpperCase();
            item.innerHTML = `
                <div class="splat-item__info">
                    <span class="splat-item__name">${splat.filename}</span>
                    <span class="splat-item__meta">${splat.folder} · ${ext} · ${splat.size_mb} MB</span>
                </div>
                <button class="btn btn-open-splat">▶ Open in LichtFeld</button>
            `;
            splatList.appendChild(item);
        });
    } catch (err) {
        console.warn('Unable to fetch splats', err);
        splatList.innerHTML = '<div class="splat-empty">Error loading splats.</div>';
    }
}
loadSplats();

splatList.addEventListener('click', (e) => {
    const btn = e.target.closest('.btn-open-splat');
    if (!btn) return;
    const item = btn.closest('.splat-item');
    openSplat(item.dataset.path);
});

async function openSplat(path) {
    const formData = new FormData();
    formData.append('path', path);
    try {
        const response = await fetch('/preview-splat', { method: 'POST', body: formData });
        const result = await response.json();
        if (result.error) {
            alert(`Failed to open splat: ${result.error}`);
        }
    } catch (err) {
        console.error(err);
        alert('Error opening splat. Check console.');
    }
}

refreshSplatsBtn.addEventListener('click', loadSplats);

// ==========================================
// Library tabs
// ==========================================
document.querySelectorAll('.library-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.library-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.library-tab__panel').forEach(p => p.style.display = 'none');
        tab.classList.add('active');
        document.getElementById(`tab-${tab.dataset.tab}`).style.display = 'block';
    });
});

// ==========================================
// Library toggle (show/hide on demand)
// ==========================================
libraryToggleBtn.addEventListener('click', () => {
    const hidden = libraryCard.style.display === 'none';
    libraryCard.style.display = hidden ? 'block' : 'none';
    libraryToggleBtn.textContent = hidden ? '✖ Hide Previous Projects' : '📚 Previous Projects';
    if (hidden) {
        loadProjects();
        loadSplats();
    }
});

// ==========================================
// Update modal
// ==========================================
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
    updateList.innerHTML = '';

    const toolUpdates = updates.filter(u => u.key === 'colmap' || u.key === 'brush' || u.key === 'sharp' || u.key === 'lichtfeld');
    const appUpdates = updates.filter(u => u.key === 'app');

    if (toolUpdates.length > 0) {
        const toolHeader = document.createElement('div');
        toolHeader.className = 'update-section-header';
        toolHeader.textContent = 'Tools (COLMAP, Brush, LichtFeld)';
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

    updateModal.style.display = 'flex';
}

function hideUpdateModal() {
    updateModal.style.display = 'none';
    updateConfirmBtn.disabled = false;
    updateConfirmBtn.textContent = 'Update Selected';
    updateCancelBtn.style.display = 'inline-block';
}

function getSelectedUpdates() {
    const checkboxes = updateList.querySelectorAll('.update-checkbox:checked');
    const selectedKeys = Array.from(checkboxes).map(cb => cb.dataset.key);
    return pendingUpdates.filter(u => selectedKeys.includes(u.key));
}

updateCancelBtn.addEventListener('click', hideUpdateModal);

updateConfirmBtn.addEventListener('click', async () => {
    const selectedUpdates = getSelectedUpdates();
    if (selectedUpdates.length === 0) {
        alert('Please select at least one component to update.');
        return;
    }

    updateConfirmBtn.disabled = true;
    updateConfirmBtn.textContent = 'Updating...';
    updateCancelBtn.style.display = 'none';

    updateList.innerHTML = `
        <div class="update-progress">
            <span class="spinner"></span>
            Installing ${selectedUpdates.length} update(s)... Check console for progress.
        </div>
    `;

    statusDiv.style.display = 'flex';
    pipelineProgress.style.display = 'block';
    advancedInfo.open = true;
    progressNote.textContent = 'Installing updates...';
    progressFill.style.width = '0%';
    progressEta.textContent = '';
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

setTimeout(checkForUpdates, 1000);

// ==========================================
// Resume Training
// ==========================================
resumeForceScratch.addEventListener('change', () => {
    if (resumeForceScratch.checked) {
        resumeCheckpoint.disabled = true;
        resumeCheckpoint.innerHTML = '<option value="">-- Ignored (Starting from Scratch) --</option>';
        resumeTrainingBtn.textContent = '▶ Start Training (Scratch)';
    } else {
        resumeCheckpoint.disabled = false;
        resumeProject.dispatchEvent(new Event('change'));
        resumeTrainingBtn.textContent = '▶ Resume Training';
    }
});

resumeProject.addEventListener('change', () => {
    const idx = parseInt(resumeProject.value, 10);

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

    const lastCheckpoint = proj.ply_checkpoints[proj.ply_checkpoints.length - 1];
    resumeCheckpoint.value = lastCheckpoint.iteration;
    brushSteps.value = lastCheckpoint.iteration + 5000;
});

refreshProjectsBtn.addEventListener('click', loadProjects);

resumeTrainingBtn.addEventListener('click', async () => {
    const projIdx = parseInt(resumeProject.value, 10);
    let startIter = parseInt(resumeCheckpoint.value, 10);
    const targetSteps = parseInt(brushSteps.value, 10);
    const forceScratch = resumeForceScratch.checked;

    if (isNaN(projIdx) || !availableProjects[projIdx]) {
        alert('Please select a project first.');
        return;
    }

    if (!forceScratch && isNaN(startIter)) {
        alert('Please select a checkpoint to resume from.');
        return;
    }

    if (forceScratch) {
        startIter = 0;
    }

    if (isNaN(targetSteps) || targetSteps <= startIter) {
        alert(`Target steps (${targetSteps}) must be greater than start iteration (${startIter}).`);
        return;
    }

    const proj = availableProjects[projIdx];

    statusDiv.style.display = 'flex';
    consoleOutput.innerHTML = '';
    initProgress({ resume: true, startIter, targetSteps });

    updateStep('training');

    const formData = new FormData();
    formData.append('projectPath', proj.path);
    formData.append('startIter', startIter);
    formData.append('totalSteps', targetSteps);
    formData.append('forceScratch', forceScratch);

    const brushSettings = {
        trainer: trainerType ? trainerType.value : "brush",
        with_viewer: brushViewer.checked,
        shutdown_after_training: brushShutdown.checked,
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

// ==========================================
// Pipeline progress & ETA estimation
// ==========================================
const STAGE_NOTES = ['Preparing inputs...', 'Searching camera placements...', 'Training splats...'];

const progress = {
    active: false,
    stageIndex: 0,        // 0 = preprocess, 1 = tracking, 2 = training
    stageStart: 0,
    estimates: [0, 0, 0], // estimated seconds per stage
    actuals: [0, 0, 0],   // measured seconds per completed stage
    total: 0,
    correction: 1,        // scales remaining estimates after a stage completes
    training: null,       // { current, total, tps, time } when step progress is parsed
    preprocessPct: null   // direct % from "Copied X/Y images (Z%)" logs
};

let progressTimer = null;

function formatEta(sec) {
    if (!isFinite(sec) || sec <= 0) return 'Estimating...';
    if (sec < 60) return `~${Math.max(1, Math.round(sec))}s left`;
    if (sec < 3600) return `~${Math.round(sec / 60)} min left`;
    return `~${(sec / 3600).toFixed(1)} hrs left`;
}

function initProgress(opts = {}) {
    let n;
    let isVideo = false;
    if (!opts.resume) {
        isVideo = selectedFiles.length > 0 && isVideoFile(selectedFiles[0]);
        if (isVideo) {
            const duration = videoDuration > 0 ? videoDuration : 60;
            n = Math.max(10, Math.round(duration * (autoExtractionFps || 2)));
        } else {
            n = Math.max(2, selectedFiles.length);
        }
    }

    const steps = opts.resume
        ? Math.max(1, (opts.targetSteps || 30000) - (opts.startIter || 0))
        : (parseInt(brushSteps.value, 10) || 30000);

    const quality = colmapQuality ? colmapQuality.value : 'high';
    const matcher = colmapMatcher ? colmapMatcher.value : 'auto';
    const engine = colmapEngine ? colmapEngine.value : 'glomap';

    const fePerImage = { low: 0.8, medium: 1.6, high: 2.6 }[quality] || 2.6;
    const pairs = (n * (n - 1)) / 2;
    const pairTime = (matcher === 'sequential' ? 0.01 : 0.03) * (quality === 'high' ? 1 : 0.6);
    const mapTime = engine === 'glomap' ? 150 : 420;
    const dupTime = (colmapRemoveDuplicates && colmapRemoveDuplicates.checked) ? n * 0.05 : 0;

    const preprocess = isVideo ? 60 + (videoDuration > 0 ? videoDuration : 60) * 6 : 20 + n * 0.15 + dupTime;
    const tracking = 45 + n * fePerImage + pairs * pairTime + mapTime;
    const training = steps * 0.035 + 90;

    Object.assign(progress, {
        active: true,
        stageIndex: opts.resume ? 2 : 0,
        stageStart: Date.now(),
        estimates: opts.resume ? [0, 0, training] : [preprocess, tracking, training],
        actuals: [0, 0, 0],
        total: opts.resume ? training : preprocess + tracking + training,
        correction: 1,
        training: null,
        preprocessPct: null
    });

    pipelineProgress.style.display = 'block';
    advancedInfo.open = false;
    progressFill.classList.remove('done', 'failed');
    progressFill.style.width = '0%';
    progressPercent.textContent = '0%';
    progressNote.textContent = opts.resume ? 'Training splats...' : 'Preparing inputs...';
    progressEta.textContent = formatEta(progress.total);

    if (progressTimer) clearInterval(progressTimer);
    progressTimer = setInterval(tickProgress, 1000);
}

function enterStage(stage) {
    if (!progress.active || stage <= progress.stageIndex) return;

    const now = Date.now();
    progress.actuals[progress.stageIndex] = (now - progress.stageStart) / 1000;

    let estDone = 0, actDone = 0;
    for (let i = 0; i <= progress.stageIndex; i++) {
        estDone += progress.estimates[i];
        actDone += progress.actuals[i];
    }
    progress.correction = estDone > 0 ? Math.min(3, Math.max(0.3, actDone / estDone)) : 1;

    progress.stageIndex = stage;
    progress.stageStart = now;
    progress.preprocessPct = null;
    if (stage === 2) progress.training = null;
    progressNote.textContent = STAGE_NOTES[stage];
}

function detectStage(msg) {
    const m = msg.toLowerCase();
    if (m.includes('step 1') || m.includes('extracting frames') || m.includes('organizing') || m.includes('copied')) return 0;
    if (m.includes('step 2') || m.includes('colmap') || m.includes('feature extraction') || m.includes('matching') || m.includes('calibrat') || m.includes('mapper')) return 1;
    if (m.includes('step 3') || m.includes('brush') || m.includes('lichtfeld') || m.includes('2dgs') || m.includes('training') || m.includes('resuming') || m.includes('scratch')) return 2;
    return -1;
}

function noteForLog(msg) {
    const m = msg.toLowerCase();
    if (m.includes('copied')) return null;
    if (m.includes('extracting frames')) return 'Extracting sharp frames...';
    if (m.includes('organizing')) return 'Organizing images...';
    if (m.includes('duplicate')) return 'Checking for duplicate images...';
    if (m.includes('feature extraction')) return 'Extracting image features...';
    if (m.includes('matching')) return 'Matching image features...';
    if (m.includes('calibrat')) return 'Calibrating cameras...';
    if (m.includes('mapper') || m.includes('colmap')) return 'Searching camera placements...';
    if (m.includes('dense')) return 'Building dense reconstruction...';
    if (m.includes('brush') || m.includes('lichtfeld') || m.includes('2dgs') || m.includes('resuming') || m.includes('scratch')) return 'Training splats...';
    return null;
}

function onTrainingStep(current, total) {
    const now = Date.now();
    const prev = progress.training;
    let tps = prev ? prev.tps : null;
    if (prev && prev.current < current) {
        const dt = (now - prev.time) / 1000;
        const dCur = current - prev.current;
        if (dt > 0.5 && dCur > 0) {
            const instant = dt / dCur;
            tps = tps ? (tps + instant) / 2 : instant;
        }
    }
    progress.training = { current, total, tps: tps || 0.035, time: now };
    progressNote.textContent = `Training splats... Step ${current.toLocaleString()}/${total.toLocaleString()}`;
}

function tickProgress() {
    if (!progress.active) return;

    const est = progress.estimates;
    const elapsed = (Date.now() - progress.stageStart) / 1000;
    const overrun = Math.max(0, elapsed - est[progress.stageIndex]);

    let doneSec = 0;
    for (let i = 0; i < progress.stageIndex; i++) doneSec += progress.actuals[i];

    let pct, eta;

    if (progress.stageIndex === 2 && progress.training && progress.training.total > 0) {
        const t = progress.training;
        doneSec += (t.current / t.total) * est[2];
        pct = Math.min(0.98, doneSec / Math.max(progress.total, 1));
        eta = Math.max(0, (t.total - t.current) * (t.tps || 0.035));
        progressNote.textContent = `Training splats... Step ${t.current.toLocaleString()}/${t.total.toLocaleString()}`;
    } else if (progress.stageIndex === 0 && progress.preprocessPct !== null) {
        doneSec += (progress.preprocessPct / 100) * est[0];
        pct = Math.min(0.98, doneSec / Math.max(progress.total, 1));
        let remaining = est[1] + est[2];
        remaining = Math.max(0, remaining - overrun) * progress.correction;
        eta = remaining;
    } else {
        const frac = Math.min(elapsed / Math.max(est[progress.stageIndex], 1), 0.95);
        doneSec += frac * est[progress.stageIndex];
        pct = Math.min(0.98, doneSec / Math.max(progress.total, 1));
        let remaining = 0;
        for (let i = progress.stageIndex; i < 3; i++) remaining += est[i];
        remaining = Math.max(0, remaining - overrun) * progress.correction;
        eta = remaining;
    }

    progressFill.style.width = (pct * 100).toFixed(1) + '%';
    progressPercent.textContent = Math.round(pct * 100) + '%';
    progressEta.textContent = formatEta(eta);
}

function handleProgressLog(msg) {
    if (!progress.active) return;

    const stage = detectStage(msg);
    if (stage > progress.stageIndex) enterStage(stage);

    const copyMatch = msg.match(/Copied\s+(\d+)\/(\d+)\s+images\s*\((\d+)%\)/i);
    if (copyMatch && progress.stageIndex === 0) {
        progress.preprocessPct = parseInt(copyMatch[3], 10);
        progressNote.textContent = `Copying images (${copyMatch[3]}%)...`;
        tickProgress();
        return;
    }

    const stepMatch = msg.match(/(?:step|iteration|progress)\D{0,40}?(\d{1,7})\s*\/\s*(\d{1,7})/i)
        || (/\bstep\b/i.test(msg) && msg.match(/(\d{1,7})\s*\/\s*(\d{1,7})/));
    if (stepMatch && progress.stageIndex === 2) {
        const cur = parseInt(stepMatch[1], 10);
        const tot = parseInt(stepMatch[2], 10);
        if (tot > 0 && cur <= tot) {
            onTrainingStep(cur, tot);
            tickProgress();
            return;
        }
    }

    const note = noteForLog(msg);
    if (note) progressNote.textContent = note;
    tickProgress();
}

function stopProgress(state) {
    if (!progress.active) return;
    progress.active = false;
    if (progressTimer) {
        clearInterval(progressTimer);
        progressTimer = null;
    }
    progressFill.style.width = '100%';
    progressFill.classList.add(state);
    if (state === 'done') {
        progressPercent.textContent = '100%';
        progressNote.textContent = 'Done! Splat saved to processing_output.';
    } else {
        progressPercent.textContent = '—';
        progressNote.textContent = 'Failed — open Advanced info for details.';
    }
    progressEta.textContent = '';
}

// ==========================================
// WebSocket setup
// ==========================================
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

        handleProgressLog(data.message);

        if (singleConsoleWindow && singleConsoleWindow.style.display !== 'none') {
            const singleLine = document.createElement('div');
            singleLine.className = 'console-line';
            singleLine.textContent = `> ${data.message}`;
            singleConsoleOutput.appendChild(singleLine);
            singleConsoleWindow.scrollTop = singleConsoleWindow.scrollHeight;
        }

        if (data.message.includes('Step 1')) updateStep('stacking');
        if (data.message.includes('COLMAP') || data.message.includes('Step 2')) updateStep('tracking');
        if (data.message.includes('Brush') || data.message.includes('Step 3')) updateStep('training');

        if (data.message.includes('SHARP')) {
            singleStepProcess.classList.add('active');
        }
    }

    if (data.type === 'status') {
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
            stopProgress('done');
            const line = document.createElement('div');
            line.className = 'console-line';
            line.style.color = 'var(--success)';
            line.textContent = 'DONE! Output available in processing_output folder.';
            consoleOutput.appendChild(line);
            markAllCompleted();
            loadSplats();

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
            stopProgress('failed');
            const line = document.createElement('div');
            line.className = 'console-line';
            line.style.color = 'var(--error)';
            line.textContent = 'FAILED! See logs above.';
            consoleOutput.appendChild(line);

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

// ==========================================
// File handling & workflow auto-detection
// ==========================================
function isVideoFile(file) {
    return file.type.startsWith('video/') ||
        /\.(mp4|mov|avi)$/i.test(file.name);
}

function isImageFile(file) {
    return file.type.startsWith('image/');
}

function handleFileSelection(files) {
    const allFiles = Array.from(files);
    if (allFiles.length === 0) return;

    // Split into images, videos and zips
    const zips = allFiles.filter(f => /\.zip$/i.test(f.name));
    const videos = allFiles.filter(isVideoFile);
    const images = allFiles.filter(isImageFile);

    // Single LiDAR capture zip -> lidar workflow
    if (zips.length === 1 && videos.length === 0 && images.length === 0) {
        setupLidarWorkflow(zips[0]);
        return;
    }

    // Single image -> Sharp workflow
    if (images.length === 1 && videos.length === 0) {
        setupSingleWorkflow(images[0]);
        return;
    }

    // Mixed selection: keep only images, warn about videos
    if (videos.length > 0 && images.length > 0) {
        alert('Videos and images cannot be mixed. Only the images will be processed.');
    }

    // Single video or multiple images -> full pipeline
    setupMultiWorkflow(videos, images);
}

function setupMultiWorkflow(videos, images) {
    singleWorkflow.style.display = 'none';
    multiWorkflow.style.display = 'block';
    selectedFiles = images.length > 0 ? images : videos;

    // Auto-select scenario
    const scenarioRadio = videos.length > 0
        ? document.querySelector('input[name="scenario"][value="video"]')
        : document.querySelector('input[name="scenario"][value="object"]');
    scenarioRadio.checked = true;
    applyScenario(scenarioRadio.value);

    // Drop zone -> summary
    dropZone.style.display = 'none';
    dropSummary.style.display = 'flex';

    if (videos.length > 0) {
        const video = videos[0];
        dropSummaryIcon.textContent = '🎬';
        dropSummaryText.textContent = video.name;

        videoDuration = 0;
        const vidEl = document.createElement('video');
        vidEl.preload = 'metadata';
        vidEl.onloadedmetadata = function () {
            window.URL.revokeObjectURL(vidEl.src);
            videoDuration = vidEl.duration;
            autoExtractionFps = computeAutoFps(videoDuration);
            const frames = Math.round(videoDuration * autoExtractionFps);
            const warn = frames < 100 ? ' ⚠️ Short video — fewer frames means lower quality.' : '';
            dropSummarySub.textContent = `Video detected — will extract ~${frames} sharp frames for best tracking quality.${warn}`;
        };
        vidEl.src = URL.createObjectURL(video);
    } else {
        const count = images.length;
        dropSummaryIcon.textContent = '📷';
        dropSummaryText.textContent = `${count} images`;
        dropSummarySub.textContent = 'Images detected — full scene reconstruction.';

        if (count > MAX_IMAGE_COUNT) {
            showImageCountModal(count);
        }
    }
}

function computeAutoFps(duration) {
    let fps = TARGET_FRAMES / duration;
    fps = Math.max(MIN_FPS, Math.min(MAX_FPS, fps));
    return Math.round(fps * 10) / 10;
}

// Too many images modal
function showImageCountModal(count) {
    imageCountMessage.textContent =
        `You selected ${count.toLocaleString()} images. More than ~400 makes camera tracking dramatically slower (matching time grows quadratically).`;

    const ratio = count / MAX_IMAGE_COUNT;
    const hours = Math.round(ratio * ratio * 1.5 * 10) / 10;
    imageCountEstimate.innerHTML =
        `<b>~400 images:</b> roughly 1–2 hours total<br>` +
        `<b>All ${count.toLocaleString()} images:</b> roughly ${hours}–${Math.round(hours * 2)} hours, but with full coverage and max detail`;

    imageCountModal.style.display = 'flex';
}

function hideImageCountModal() {
    imageCountModal.style.display = 'none';
}

useAllImagesBtn.addEventListener('click', () => {
    hideImageCountModal();
    startUpload();
});

useFewerImagesBtn.addEventListener('click', () => {
    hideImageCountModal();
    const step = Math.max(1, Math.ceil(selectedFiles.length / MAX_IMAGE_COUNT));
    const sampled = selectedFiles.filter((_, i) => i % step === 0);
    const originalCount = selectedFiles.length;
    selectedFiles = sampled;
    dropSummarySub.textContent = `Reduced from ${originalCount.toLocaleString()} to ${sampled.length.toLocaleString()} images (evenly sampled for speed).`;
    startUpload();
});

// ==========================================
// Multi workflow upload
// ==========================================
startUploadBtn.addEventListener('click', startUpload);

async function startUpload() {
    if (!selectedFiles.length) {
        alert("Select images or a video first.");
        return;
    }

    const isVideo = isVideoFile(selectedFiles[0]);

    // UI Updates
    uploadCard.style.display = 'none';
    multiWorkflow.style.display = 'none';
    statusDiv.style.display = 'flex';
    consoleOutput.innerHTML = '';
    initProgress();

    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files', file);
    });

    if (isVideo) {
        formData.append('extractionMode', 'fps');
        formData.append('extractionValue', autoExtractionFps);
        formData.append('blurFilter', 'true');
    }

    const projectName = projectNameInput.value.trim();
    if (projectName) {
        formData.append('projectName', projectName);
    }

    const colmapSettings = {
        engine: colmapEngine.value,
        matcher: colmapMatcher.value,
        quality: colmapQuality.value,
        dense: colmapDense.checked,
        remove_duplicates: colmapRemoveDuplicates.checked
    };

    const brushSettings = {
        trainer: trainerType ? trainerType.value : "brush",
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

// ==========================================
// LiDAR capture workflow
// ==========================================
function setupLidarWorkflow(zipFile) {
    multiWorkflow.style.display = 'none';
    singleWorkflow.style.display = 'none';
    lidarWorkflow.style.display = 'block';
    selectedFiles = [];
    lidarZipFile = zipFile;

    dropZone.style.display = 'none';
    dropSummary.style.display = 'none';

    lidarZipName.textContent = zipFile.name;
    if (!lidarProjectName.value) {
        const base = zipFile.name.replace(/\.zip$/i, '').replace(/[^a-zA-Z0-9_\-]/g, '_');
        lidarProjectName.value = base.slice(0, 64);
    }

    applyLidarScenario(document.querySelector('input[name="lidarScenario"]:checked').value);
}

lidarChangeBtn.addEventListener('click', () => {
    lidarWorkflow.style.display = 'none';
    dropZone.style.display = 'block';
    dropSummary.style.display = 'none';
    lidarZipFile = null;
    fileInput.value = '';
});

startLidarBtn.addEventListener('click', () => {
    if (!lidarZipFile) {
        alert('Select a LiDAR capture ZIP first.');
        return;
    }
    startLidarUpload(lidarZipFile);
});

async function startLidarUpload(zipFile) {
    lidarWorkflow.style.display = 'none';
    uploadCard.style.display = 'none';
    statusDiv.style.display = 'flex';
    consoleOutput.innerHTML = '';

    // LiDAR pipeline skips COLMAP mapping; estimate only training time.
    initProgress({ resume: true, startIter: 0, targetSteps: parseInt(lidarBrushSteps.value, 10) || 50000 });

    const formData = new FormData();
    formData.append('file', zipFile);

    const projectName = lidarProjectName.value.trim();
    if (projectName) {
        formData.append('projectName', projectName);
    }

    const colmapSettings = {
        refine_poses: lidarRefinePoses.checked
    };
    formData.append('colmapSettings', JSON.stringify(colmapSettings));

    const brushSettings = {
        trainer: lidarTrainerType.value,
        total_steps: parseInt(lidarBrushSteps.value, 10) || 50000,
        with_viewer: lidarViewer.checked,
        shutdown_after_training: lidarShutdown.checked,
        sh_degree: 3,
        max_splats: parseInt(lidarMaxSplats.value, 10) || 5000000,
        max_resolution: 8192
    };
    formData.append('brushSettings', JSON.stringify(brushSettings));

    try {
        const response = await fetch('/upload-lidar', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Upload failed');

        const result = await response.json();
        console.log('LiDAR task started:', result.task_id);
    } catch (err) {
        console.error(err);
        alert('Error uploading LiDAR capture. Check console.');
        resetLidarWorkflow();
    }
}

function resetLidarWorkflow() {
    uploadCard.style.display = 'block';
    dropZone.style.display = 'block';
    lidarWorkflow.style.display = 'none';
    lidarZipFile = null;
}

// ==========================================
// Single image workflow
// ==========================================
function setupSingleWorkflow(file) {
    multiWorkflow.style.display = 'none';
    singleWorkflow.style.display = 'block';
    singleImageFile = file;

    dropZone.style.display = 'none';
    dropSummary.style.display = 'none';

    // Preview thumbnail
    singlePreviewPlaceholder.style.display = 'none';
    singlePreviewImg.style.display = 'block';
    singlePreviewImg.src = URL.createObjectURL(file);
}

singleChangeBtn.addEventListener('click', () => {
    singleWorkflow.style.display = 'none';
    dropZone.style.display = 'block';
    dropSummary.style.display = 'none';
    singleImageFile = null;
    singlePreviewImg.src = '';
    singlePreviewImg.style.display = 'none';
    singlePreviewPlaceholder.style.display = 'flex';
});

startSingleBtn.addEventListener('click', () => {
    if (!singleImageFile) {
        alert('Select an image first.');
        return;
    }
    startSingleUpload(singleImageFile);
});

async function startSingleUpload(file) {
    singleWorkflow.style.display = 'none';
    uploadCard.style.display = 'none';
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
    uploadCard.style.display = 'block';
    dropZone.style.display = 'block';
    singleWorkflow.style.display = 'none';
    singlePipelineStatus.style.display = 'none';
    singleConsoleWindow.style.display = 'none';
    singleStepProcess.classList.remove('active', 'completed');
}

// ==========================================
// Drag and drop
// ==========================================
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

changeFilesBtn.addEventListener('click', () => {
    multiWorkflow.style.display = 'none';
    singleWorkflow.style.display = 'none';
    lidarWorkflow.style.display = 'none';
    dropZone.style.display = 'block';
    dropSummary.style.display = 'none';
    selectedFiles = [];
    lidarZipFile = null;
    singleImageFile = null;
    fileInput.value = '';
});
