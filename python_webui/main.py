from fastapi import FastAPI, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from typing import List, Optional
import shutil
import uuid
import json
import asyncio
from pathlib import Path
from utils import logger, ensure_directory, get_project_root
from pipeline_manager import PipelineManager
from update_manager import get_update_manager

import sys

# Force ProactorEventLoop on Windows for subprocess support
# This must be done before the event loop is created
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

app = FastAPI(title="Gaussian Splatter WebUI")

# Paths
BASE_DIR = get_project_root()
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "processing_output"

ensure_directory(UPLOAD_DIR)
ensure_directory(OUTPUT_DIR)

# Cleanup backup folders on startup
# The backup folder is expected to be in the project root, which is one level up from python_webui
backup_dir = BASE_DIR.parent / "colmap-x64-windows-cuda_backup"
if backup_dir.exists() and backup_dir.is_dir():
    try:
        shutil.rmtree(backup_dir)
        logger.info(f"Removed backup directory: {backup_dir}")
    except Exception as e:
        logger.error(f"Failed to remove backup directory {backup_dir}: {e}")

# Mount static files
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Pipeline Manager
manager = PipelineManager(str(OUTPUT_DIR))

# Store active websockets for logging
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        if websocket in self.active_connections:
            try:
                await websocket.send_text(message)
            except:
                pass

ws_manager = ConnectionManager()

@app.get("/")
async def get_index():
    with open(BASE_DIR / "templates" / "index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.post("/upload-single")
async def upload_single_image(
    file: UploadFile = File(...),
    device: str = Form("cuda"),
    render: str = Form("true"),
    projectName: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    """Upload a single image for Sharp 3DGS generation."""
    try:
        task_id = str(uuid.uuid4())
        
        # Create output directory
        project_folder = projectName if projectName else task_id[:8]
        output_path = OUTPUT_DIR / f"sharp_{project_folder}"
        ensure_directory(output_path)
        
        # Save uploaded image
        input_path = output_path / file.filename
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Single image uploaded: {file.filename} -> {output_path}")
        
        # Parse render setting
        should_render = render.lower() == "true"
        
        # Start Sharp processing in background
        background_tasks.add_task(
            start_sharp_pipeline,
            task_id,
            str(input_path),
            str(output_path),
            device,
            should_render
        )
        
        return {"task_id": task_id, "status": "uploaded", "message": "Sharp processing started"}
    except Exception as e:
        logger.error(f"Single image upload failed: {e}")
        return {"error": str(e)}

async def start_sharp_pipeline(task_id: str, input_path: str, output_path: str, device: str, render: bool):
    """Background task to run Sharp model."""
    async def log_callback(msg: str):
        try:
            await ws_manager.broadcast(json.dumps({
                "type": "log",
                "task_id": task_id,
                "message": msg
            }))
        except Exception as e:
            logger.error(f"Failed to broadcast log: {e}")

    try:
        await manager.run_sharp(input_path, output_path, device, render, log_callback)
        await log_callback("Sharp processing finished successfully!")
        await ws_manager.broadcast(json.dumps({
            "type": "status",
            "task_id": task_id,
            "status": "completed"
        }))
    except Exception as e:
        logger.exception(f"Sharp pipeline failed for task {task_id}")
        await ws_manager.broadcast(json.dumps({
            "type": "status",
            "task_id": task_id,
            "status": "failed"
        }))


@app.post("/upload")
async def upload_dataset(
    files: List[UploadFile] = File(...), 
    extractionMode: str = Form("fps"),
    extractionValue: str = Form("2"),
    projectName: Optional[str] = Form(None),
    colmapSettings: Optional[str] = Form(None),
    brushSettings: Optional[str] = Form(None),
    background_tasks: BackgroundTasks = None
):
    try:
        task_id = str(uuid.uuid4())
        task_dir = UPLOAD_DIR / task_id
        ensure_directory(task_dir)
        
        input_type = "images"
        input_path = task_dir 
        
        # Check if single video file
        if len(files) == 1 and files[0].filename.lower().endswith(('.mp4', '.mov', '.avi')):
            input_type = "video"
            input_path = task_dir / files[0].filename
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(files[0].file, buffer)
        else:
            # Assume images
            input_type = "images"
            images_dir = task_dir / "raw_images"
            ensure_directory(images_dir)
            input_path = images_dir
            
            for file in files:
                if file.filename:
                    file_path = images_dir / file.filename
                    with open(file_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"Files uploaded: {task_id} (Type: {input_type})")
        
        extraction_settings = {
            "mode": extractionMode,
            "value": extractionValue
        }

        # Optional pipeline settings (provided as JSON strings)
        parsed_colmap = None
        parsed_brush = None

        try:
            if colmapSettings:
                parsed_colmap = json.loads(colmapSettings)
        except Exception as e:
            logger.error(f"Failed to parse colmap settings: {e}")

        try:
            if brushSettings:
                parsed_brush = json.loads(brushSettings)
        except Exception as e:
            logger.error(f"Failed to parse brush settings: {e}")

        # Start processing in background
        background_tasks.add_task(
            start_pipeline,
            task_id,
            input_type,
            input_path,
            extraction_settings,
            parsed_colmap,
            parsed_brush,
            projectName
        )
        
        return {"task_id": task_id, "status": "uploaded", "message": "File processing started"}
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return {"error": str(e)}

@app.get("/settings")
async def get_settings():
    """Expose default COLMAP and Brush settings to the UI."""
    return {
        "colmap": manager.get_default_colmap_settings(),
        "brush": manager.get_default_brush_settings()
    }

@app.get("/list-outputs")
async def list_outputs():
    """List available output folders that can be used for resume training."""
    try:
        outputs = manager.get_available_outputs()
        logger.info(f"List outputs called. Found {len(outputs)} projects: {[o['folder'] for o in outputs]}")
        return {"outputs": outputs}
    except Exception as e:
        logger.error(f"Failed to list outputs: {e}")
        return {"error": str(e), "outputs": []}

@app.post("/resume")
async def resume_training(
    projectPath: str = Form(...),
    startIter: Optional[int] = Form(0), # Optional now because scratch might not send it or it's irrelevant
    totalSteps: int = Form(...),
    brushSettings: Optional[str] = Form(None),
    forceScratch: bool = Form(False),
    background_tasks: BackgroundTasks = None
):
    """Resume training from an existing project folder or start from scratch."""
    try:
        # Parse brush settings if provided
        parsed_brush = None
        if brushSettings:
            try:
                parsed_brush = json.loads(brushSettings)
            except Exception as e:
                logger.error(f"Failed to parse brush settings: {e}")
        
        # Ensure total_steps is set
        if parsed_brush is None:
            parsed_brush = {}
        parsed_brush["total_steps"] = totalSteps
        
        msg = f"Resuming training from iteration {startIter}"
        if forceScratch:
            msg = "Starting training from scratch using existing COLMAP data"
        
        # Start resume training in background
        background_tasks.add_task(
            start_resume_training,
            projectPath,
            startIter,
            parsed_brush,
            forceScratch
        )
        
        return {"status": "started", "message": msg}
    except Exception as e:
        logger.error(f"Training request failed: {e}")
        return {"error": str(e)}

async def start_resume_training(project_path: str, start_iter: int, brush_settings: dict, force_scratch: bool):
    """Background task to run resume training."""
    async def log_callback(msg: str):
        try:
            await ws_manager.broadcast(json.dumps({
                "type": "log",
                "task_id": "resume",
                "message": msg
            }))
        except Exception as e:
            logger.error(f"Failed to broadcast log: {e}")

    try:
        await manager.resume_training(project_path, start_iter, brush_settings, log_callback, force_scratch)
        await ws_manager.broadcast(json.dumps({
            "type": "status",
            "task_id": "resume",
            "status": "completed"
        }))
    except Exception as e:
        logger.exception(f"Resume training failed")
        await ws_manager.broadcast(json.dumps({
            "type": "status",
            "task_id": "resume",
            "status": "failed"
        }))

async def start_pipeline(task_id: str, input_type: str, input_path: Path, extraction_settings: dict, colmap_settings: Optional[dict] = None, brush_settings: Optional[dict] = None, project_name: Optional[str] = None):
    async def log_callback(msg: str):
        try:
            await ws_manager.broadcast(json.dumps({
                "type": "log",
                "task_id": task_id,
                "message": msg
            }))
        except Exception as e:
            logger.error(f"Failed to broadcast log: {e}")

    try:
        await manager.process_dataset(
            task_id,
            input_type,
            input_path,
            extraction_settings,
            log_callback,
            colmap_settings,
            brush_settings,
            project_name
        )
        await log_callback("Pipeline finished successfully") # Explicit log
        await ws_manager.broadcast(json.dumps({
            "type": "status",
            "task_id": task_id,
            "status": "completed"
        }))
    except Exception as e:
        logger.exception(f"Pipeline failed for task {task_id}")
        await ws_manager.broadcast(json.dumps({
            "type": "status",
            "task_id": task_id,
            "status": "failed"
        }))

@app.get("/check-updates")
async def check_updates():
    """Check for available updates from GitHub."""
    try:
        update_manager = get_update_manager()
        result = update_manager.check_for_updates()
        return result
    except Exception as e:
        logger.error(f"Failed to check updates: {e}")
        return {"updates_available": False, "updates": [], "error": str(e)}

@app.post("/install-updates")
async def install_updates(
    updates: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """Install selected updates."""
    try:
        updates_list = json.loads(updates)
        
        # Start update installation in background
        background_tasks.add_task(
            run_update_installation,
            updates_list
        )
        
        return {"status": "started", "message": "Update installation started"}
    except Exception as e:
        logger.error(f"Failed to start update installation: {e}")
        return {"error": str(e)}

async def run_update_installation(updates_list: list):
    """Background task to install updates."""
    update_manager = get_update_manager()
    
    async def log_callback(msg: str):
        try:
            await ws_manager.broadcast(json.dumps({
                "type": "log",
                "task_id": "update",
                "message": msg
            }))
        except Exception as e:
            logger.error(f"Failed to broadcast update log: {e}")
    
    all_success = True
    for update in updates_list:
        app_key = update.get("key")
        download_url = update.get("download_url", "")
        latest_version = update.get("latest", "")
        
        await log_callback(f"Installing update for {update.get('name', app_key)}...")
        
        # Run in executor since it's blocking
        import asyncio
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(
            None,
            lambda: update_manager.install_update(
                app_key,
                download_url,
                latest_version,
                lambda msg: asyncio.run_coroutine_threadsafe(log_callback(msg), loop)
            )
        )
        
        if not success:
            all_success = False
            await log_callback(f"Failed to update {update.get('name', app_key)}")
    
    if all_success:
        await ws_manager.broadcast(json.dumps({
            "type": "status",
            "task_id": "update",
            "status": "completed"
        }))
    else:
        await ws_manager.broadcast(json.dumps({
            "type": "status",
            "task_id": "update",
            "status": "partial"
        }))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming messages if needed
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
