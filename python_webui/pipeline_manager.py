import asyncio
import os
from pathlib import Path
from typing import List, Callable, Optional
import shutil
import re
from utils import logger, ensure_directory

import cv2

# Hardcoded paths based on user environment
COLMAP_BAT_PATH = r"d:\software\gaussian_splatter\colmap-x64-windows-cuda\COLMAP.bat"
BRUSH_PATH = r"d:\software\gaussian_splatter\brush-app-x86_64-pc-windows-msvc\brush_app.exe"

DEFAULT_COLMAP_SETTINGS = {
    "sparse": 1,      # Build sparse model
    "dense": 0,       # Skip dense reconstruction for speed by default
    "quality": "high"
}

DEFAULT_BRUSH_SETTINGS = {
    "total_steps": 30000,
    "with_viewer": True,
    "sh_degree": 3,           # Spherical Harmonics degree (0-3)
    "max_splats": 3000000,    # Max 3 million splats
    "max_resolution": 8192    # Max resolution (limited by WebGPU dispatch group max of 65535)
}

class PipelineManager:
    def __init__(self, base_output_dir: str = "processing_output"):
        self.base_output_dir = Path(base_output_dir)
        ensure_directory(self.base_output_dir)
        self.active_tasks = {}

    def get_default_colmap_settings(self):
        return DEFAULT_COLMAP_SETTINGS.copy()

    def get_default_brush_settings(self):
        return DEFAULT_BRUSH_SETTINGS.copy()

    def _merge_settings(self, user_settings: Optional[dict], defaults: dict) -> dict:
        merged = defaults.copy()
        if user_settings:
            for key, value in user_settings.items():
                if value is not None:
                    merged[key] = value
        return merged

    def _build_colmap_args(self, colmap_dir: Path, images_dir: Path, settings: dict) -> List[str]:
        return [
            "automatic_reconstructor",
            f"--workspace_path \"{colmap_dir}\"",
            f"--image_path \"{images_dir}\"",
            f"--sparse {1 if settings.get('sparse', 1) else 0}",
            f"--dense {1 if settings.get('dense', 0) else 0}",
            f"--quality {settings.get('quality', 'medium')}"
        ]

    def _build_brush_args(self, task_dir: Path, model_dir: Path, settings: dict) -> List[str]:
        args = [
            f"\"{task_dir}\"",
            f"--total-steps {int(settings.get('total_steps', 30000))}",
            f"--sh-degree {int(settings.get('sh_degree', 3))}",
            f"--max-splats {int(settings.get('max_splats', 3000000))}",
            f"--max-resolution {int(settings.get('max_resolution', 8192))}",
            f"--export-path \"{model_dir}\""
        ]
        if settings.get("with_viewer", True):
            args.append("--with-viewer")
        return args

    def _build_resume_brush_args(self, task_dir: Path, model_dir: Path, start_iter: int, settings: dict) -> List[str]:
        """Build Brush args with --start-iter for resuming training."""
        args = [
            f"\"{task_dir}\"",
            f"--total-steps {int(settings.get('total_steps', 30000))}",
            f"--start-iter {int(start_iter)}",
            f"--sh-degree {int(settings.get('sh_degree', 3))}",
            f"--max-splats {int(settings.get('max_splats', 3000000))}",
            f"--max-resolution {int(settings.get('max_resolution', 8192))}",
            f"--export-path \"{model_dir}\""
        ]
        if settings.get("with_viewer", True):
            args.append("--with-viewer")
        return args

    def get_available_outputs(self) -> List[dict]:
        """List all output folders that have valid sparse data for resume training."""
        outputs = []
        if not self.base_output_dir.exists():
            return outputs
        
        for folder in self.base_output_dir.iterdir():
            if folder.is_dir():
                sparse_path = folder / "sparse"
                model_path = folder / "model"
                
                if sparse_path.exists():
                    # Find available PLY checkpoints
                    ply_files = []
                    if model_path.exists():
                        for ply in model_path.glob("export_*.ply"):
                            # Extract iteration number from filename
                            try:
                                iter_str = ply.stem.replace("export_", "")
                                iteration = int(iter_str)
                                ply_files.append({
                                    "filename": ply.name,
                                    "iteration": iteration
                                })
                            except ValueError:
                                pass
                    
                    ply_files.sort(key=lambda x: x["iteration"])
                    
                    outputs.append({
                        "folder": folder.name,
                        "path": str(folder),
                        "has_sparse": True,
                        "ply_checkpoints": ply_files
                    })
        
        return outputs

    async def resume_training(self, project_path: str, start_iter: int, brush_settings: dict, log_callback: Callable[[str], None], force_scratch: bool = False):
        """
        Resume training from an existing project folder OR start from scratch using existing sparse data.
        
        Args:
            project_path: Path to existing project folder with sparse/ data
            start_iter: Iteration to resume from (must match an exported PLY) - Ignored if force_scratch is True
            brush_settings: Brush training settings including total_steps target
            log_callback: Callback for streaming logs
            force_scratch: If True, restarts training from step 0 using sparse data
        """
        task_dir = Path(project_path)
        model_dir = task_dir / "model"
        sparse_path = task_dir / "sparse"
        
        # Validate paths
        if not task_dir.exists():
            raise Exception(f"Project folder not found: {task_dir}")
        
        if not sparse_path.exists():
            raise Exception(f"No sparse data found in project. Cannot train without COLMAP data.")
        
        ensure_directory(model_dir)

        if force_scratch:
            # START FROM SCRATCH logic
            init_ply_path = task_dir / "init.ply"
            if init_ply_path.exists():
                await log_callback(f"Removing existing init.ply to start fresh...\n")
                init_ply_path.unlink()
            
            await log_callback(f"--- Starting Training from Scratch (using existing COLMAP data) ---\n")
            await log_callback(f"Project: {task_dir}\n")
            
            try:
                brush_cfg = self._merge_settings(brush_settings, DEFAULT_BRUSH_SETTINGS)
                # Use standard brush args (no start-iter)
                brush_args = self._build_brush_args(task_dir, model_dir, brush_cfg)
                await self.run_command(BRUSH_PATH, brush_args, log_callback)
                
                await log_callback("--- Training Completed Successfully ---\n")
            except Exception as e:
                logger.error(f"Training failed: {e}")
                await log_callback(f"\nCRITICAL ERROR: {str(e)}\n")
                raise e

        else:
            # RESUME logic
            # Find the PLY checkpoint
            expected_ply = model_dir / f"export_{start_iter}.ply"
            if not expected_ply.exists():
                available = list(model_dir.glob("export_*.ply"))
                available_iters = [p.stem.replace("export_", "") for p in available]
                raise Exception(f"PLY checkpoint for iteration {start_iter} not found. Available: {available_iters}")
            
            # CRITICAL: Copy checkpoint PLY to init.ply in project root
            # Brush uses init.ply as the initialization point for training
            init_ply_path = task_dir / "init.ply"
            await log_callback(f"Copying checkpoint {expected_ply.name} to init.ply for initialization...\n")
            shutil.copy2(expected_ply, init_ply_path)
            
            await log_callback(f"--- Resuming Training from iteration {start_iter} ---\n")
            await log_callback(f"Project: {task_dir}\n")
            await log_callback(f"Target steps: {brush_settings.get('total_steps', 30000)}\n")
            
            try:
                brush_cfg = self._merge_settings(brush_settings, DEFAULT_BRUSH_SETTINGS)
                brush_args = self._build_resume_brush_args(task_dir, model_dir, start_iter, brush_cfg)
                await self.run_command(BRUSH_PATH, brush_args, log_callback)
                
                await log_callback("--- Resume Training Completed Successfully ---\n")
            except Exception as e:
                logger.error(f"Resume training failed: {e}")
                await log_callback(f"\nCRITICAL ERROR: {str(e)}\n")
                raise e

    async def run_command(self, command: str, args: List[str], log_callback: Optional[Callable[[str], None]] = None, cwd: Optional[Path] = None):
        """Runs a shell command asynchronously and streams output."""
        full_command = f'"{command}" ' + " ".join(args)
        logger.info(f"Starting command: {full_command}")
        
        if log_callback:
            await log_callback(f"Executing: {full_command}\n")

        process = await asyncio.create_subprocess_shell(
            full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd
        )

        async def read_stream(stream, callback):
            while True:
                line = await stream.readline()
                if line:
                    try:
                        decoded = line.decode('utf-8').strip()
                    except UnicodeDecodeError:
                        try:
                            decoded = line.decode('cp1252', errors='replace').strip()
                        except:
                            decoded = line.decode('utf-8', errors='replace').strip()
                            
                    logger.info(f"CMD OUT: {decoded}")
                    if callback:
                        await callback(f"{decoded}\n")
                else:
                    break

        await asyncio.gather(
            read_stream(process.stdout, log_callback),
            read_stream(process.stderr, log_callback)
        )

        return_code = await process.wait()
        if return_code != 0:
            error_msg = f"Command failed with exit code {return_code}"
            logger.error(error_msg)
            if log_callback:
                await log_callback(f"ERROR: {error_msg}\n")
            raise Exception(error_msg)
        
        logger.info("Command finished successfully")
        if log_callback:
            await log_callback("Command finished successfully\n")

    def extract_frames(self, video_path: Path, output_dir: Path, settings: dict, log_callback: Callable[[str], None]):
        """Extracts frames from video based on settings."""
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception("Failed to open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / orig_fps

        target_mode = settings.get('mode', 'fps') # 'fps' or 'count'
        
        if target_mode == 'count':
            target_count = int(settings.get('value', 100))
            # Calculate skip interval to get approx target_count
            step = max(1, total_frames / target_count)
        else: # 'fps'
            target_fps = float(settings.get('value', 2))
            # If target fps is higher than video fps, take all frames
            if target_fps >= orig_fps:
                step = 1
            else:
                step = orig_fps / target_fps
        
        logger.info(f"Video Stats: {total_frames} frames, {orig_fps} fps, {duration:.2f}s")
        logger.info(f"Extraction: mode={target_mode}, step={step:.2f}")

        # Sync wrapper for async callback if needed, but here we run blocking 
        # since it's inside an async wrapper in main usually, or we can just print for now
        # Ideally this should be run in a separate thread if blocking loop.
        
        count = 0
        saved_count = 0
        next_frame_to_save = 0.0
        
        ensure_directory(output_dir)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if count >= next_frame_to_save:
                frame_name = f"frame_{saved_count:05d}.jpg"
                cv2.imwrite(str(output_dir / frame_name), frame)
                saved_count += 1
                next_frame_to_save += step
            
            count += 1

        cap.release()
        return saved_count, orig_fps

    async def process_dataset(self, task_id: str, input_type: str, input_path: Path, extraction_settings: dict, log_callback: Callable[[str], None], colmap_settings: Optional[dict] = None, brush_settings: Optional[dict] = None, project_name: Optional[str] = None):
        """
        Full pipeline:
        1. Preprocessing (Video Extraction or Image Organization) -> Output to 'images' folder
        2. COLMAP (Tracking) -> Output to 'sparse' folder
        3. Brush (Training) -> Output .ply file
        
        Args:
            task_id: Unique task identifier
            input_type: 'video' or 'images'
            input_path: Path to input files
            extraction_settings: Video extraction settings
            log_callback: Callback for streaming logs
            colmap_settings: Optional COLMAP settings override
            brush_settings: Optional Brush settings override
            project_name: Optional custom folder name for output (sanitized)
        """
        # Determine the output folder name
        if project_name and project_name.strip():
            # Sanitize the project name: only allow alphanumeric, underscore, hyphen
            sanitized_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', project_name.strip())
            # Avoid empty name after sanitization
            if not sanitized_name:
                sanitized_name = task_id
            
            # Check if folder exists and append counter if needed
            output_folder_name = sanitized_name
            counter = 1
            while (self.base_output_dir / output_folder_name).exists():
                output_folder_name = f"{sanitized_name}_{counter}"
                counter += 1
            
            await log_callback(f"Using project name: {output_folder_name}\n")
        else:
            output_folder_name = task_id
        
        task_dir = self.base_output_dir / output_folder_name
        ensure_directory(task_dir)
        
        # input_path is e.g. uploads/task_id/video.mp4 OR uploads/task_id/images/
        # But we want to organize everything into task_dir/images for COLMAP
        
        images_dir = task_dir / "images" 
        colmap_dir = task_dir / "colmap_workspace"
        model_dir = task_dir / "model"
        
        ensure_directory(images_dir)
        ensure_directory(colmap_dir)
        ensure_directory(model_dir)

        colmap_cfg = self._merge_settings(colmap_settings, DEFAULT_COLMAP_SETTINGS)
        brush_cfg = self._merge_settings(brush_settings, DEFAULT_BRUSH_SETTINGS)

        try:
            # 1. Preprocessing
            await log_callback("--- Step 1: Preprocessing Inputs ---\n")
            
            if input_type == 'video':
                await log_callback(f"Extracting frames from video... Mode: {extraction_settings.get('mode')} Value: {extraction_settings.get('value')}\n")
                
                # Run extraction in thread pool to avoid blocking asyncio loop
                loop = asyncio.get_running_loop()
                num_extracted, orig_fps = await loop.run_in_executor(
                    None, 
                    self.extract_frames, 
                    input_path, 
                    images_dir, 
                    extraction_settings, 
                    log_callback
                )
                
                await log_callback(f"Extracted {num_extracted} frames (Source: {orig_fps:.2f} fps).\n")
                
            else: # images
                # If images are already in a folder (uploads/task_id/raw_images/), move/copy them to images_dir
                # Assuming input_path is the directory containing images
                await log_callback("Organizing uploaded images...\n")
                
                # First, collect all image files to copy
                image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
                files_to_copy = []
                for root, dirs, files in os.walk(input_path):
                    for file in files:
                        if Path(file).suffix.lower() in image_extensions:
                            src = Path(root) / file
                            dst = images_dir / file  # Flat structure
                            files_to_copy.append((src, dst))
                
                total_files = len(files_to_copy)
                await log_callback(f"Found {total_files} images to copy...\n")
                
                # Copy files with progress updates (run in executor to avoid blocking)
                loop = asyncio.get_running_loop()
                
                def copy_files_with_progress():
                    copied = 0
                    for src, dst in files_to_copy:
                        shutil.copy2(src, dst)
                        copied += 1
                    return copied
                
                # For large batches, copy in chunks so we can report progress
                copied_count = 0
                chunk_size = 50
                for i in range(0, total_files, chunk_size):
                    chunk = files_to_copy[i:i + chunk_size]
                    
                    def copy_chunk(chunk_to_copy):
                        for src, dst in chunk_to_copy:
                            shutil.copy2(src, dst)
                        return len(chunk_to_copy)
                    
                    count = await loop.run_in_executor(None, copy_chunk, chunk)
                    copied_count += count
                    
                    # Log progress
                    progress_pct = int((copied_count / total_files) * 100)
                    await log_callback(f"Copied {copied_count}/{total_files} images ({progress_pct}%)...\n")
                
                await log_callback(f"Prepared {copied_count} images for COLMAP.\n")

            # 2. COLMAP
            await log_callback("--- Step 2: Running COLMAP (Tracking) ---\n")
            
            # Check if we have enough images
            valid_images = list(images_dir.glob("*.tif")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            if len(valid_images) < 2:
                msg = f"Insufficient images for COLMAP reconstruction. Found {len(valid_images)}. Need at least 2 (preferably 20+)."
                await log_callback(f"CRITICAL ERROR: {msg}\n")
                raise Exception(msg)

            # We use 'automatic_reconstructor' for simplicity
            database_path = colmap_dir / "database.db"
            
            colmap_args = self._build_colmap_args(colmap_dir, images_dir, colmap_cfg)
            await self.run_command(COLMAP_BAT_PATH, colmap_args, log_callback)

            # 3. Brush Training
            await log_callback("--- Step 3: Running Brush (Training) ---\n")
            # Brush needs the colmap sparse output. 
            # automatic_reconstructor usually puts sparse model in workspace_path/sparse/0
            sparse_path = colmap_dir / "sparse" / "0"
            
            if not sparse_path.exists():
                raise Exception("COLMAP did not produce a sparse model in the expected location.")
            
            final_sparse_dir = task_dir / "sparse"
            if final_sparse_dir.exists():
                shutil.rmtree(final_sparse_dir)
            ensure_directory(final_sparse_dir)

            # Find the largest sparse model in colmap_dir/sparse
            sparse_candidates = []
            colmap_sparse = colmap_dir / "sparse"
            
            # Check for subfolders (0, 1, etc.)
            for folder in colmap_sparse.iterdir():
                if folder.is_dir():
                    # Check for bin or txt model files
                    points_bin = folder / "points3D.bin"
                    points_txt = folder / "points3D.txt"
                    
                    size = 0
                    if points_bin.exists():
                        size = points_bin.stat().st_size
                    elif points_txt.exists():
                        size = points_txt.stat().st_size
                    
                    if size > 0:
                        sparse_candidates.append((size, folder))
            
            if not sparse_candidates:
                 # Fallback: maybe the sparse folder IS the model itself (flat structure)
                 points_bin = colmap_sparse / "points3D.bin"
                 points_txt = colmap_sparse / "points3D.txt"
                 if points_bin.exists() or points_txt.exists():
                     shutil.copytree(colmap_sparse, final_sparse_dir / "0")
                     logger.info("Copied flat sparse model to sparse/0")
                 else:
                     raise Exception("COLMAP did not produce a valid sparse model (points3D not found).")
            else:
                # Sort by size descending (largest points3D = most reconstructed points)
                sparse_candidates.sort(key=lambda x: x[0], reverse=True)
                best_model_path = sparse_candidates[0][1]
                
                logger.info(f"Multiple COLMAP models found. Selected largest: {best_model_path.name} (points3D size: {sparse_candidates[0][0]} bytes)")
                
                shutil.copytree(best_model_path, final_sparse_dir / "0")

            brush_args = self._build_brush_args(task_dir, model_dir, brush_cfg)
            await self.run_command(BRUSH_PATH, brush_args, log_callback)

            await log_callback("--- Pipeline Completed Successfully ---\n")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            await log_callback(f"\nCRITICAL ERROR: {str(e)}\n")
            raise e
