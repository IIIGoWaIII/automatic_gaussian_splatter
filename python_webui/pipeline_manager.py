import asyncio
import os
import zipfile
from pathlib import Path
from typing import List, Callable, Optional
import shutil
import re
import hashlib
import subprocess
from utils import logger, ensure_directory, get_project_root

import cv2
import numpy as np

# Dynamic paths based on project root
REPO_ROOT = get_project_root().parent
COLMAP_BAT_PATH = str(REPO_ROOT / "colmap-x64-windows-cuda" / "COLMAP.bat")
COLMAP_EXE_PATH = str(REPO_ROOT / "colmap-x64-windows-cuda" / "bin" / "colmap.exe")
BRUSH_PATH = str(REPO_ROOT / "brush-app-x86_64-pc-windows-msvc" / "brush_app.exe")
LICHTFELD_PATH = str(REPO_ROOT / "LichtFeld-Studio-windows-nightly-2026-02-03-5a92bff" / "bin" / "LichtFeld-Studio.exe")
SHARP_PATH = str(REPO_ROOT / "ml-sharp" / ".venv" / "Scripts" / "sharp.exe")
TWO_DGS_DIR = REPO_ROOT / "2d-gaussian-splatting-main"
TWO_DGS_PYTHON = str(TWO_DGS_DIR / ".venv" / "Scripts" / "python.exe")

DEFAULT_COLMAP_SETTINGS = {
    "engine": "glomap",        # "glomap" | "incremental"
    "matcher": "auto",         # "auto" | "exhaustive" | "sequential"
    "quality": "high",         # "low" | "medium" | "high"
    "dense": 0,                # Skip dense reconstruction for speed by default
    "remove_duplicates": False, # Remove duplicate images
}

DEFAULT_BRUSH_SETTINGS = {
    "trainer": "brush",       # 'brush' or 'lichtfeld'
    "total_steps": 30000,
    "with_viewer": True,
    "sh_degree": 3,           # Spherical Harmonics degree (0-3)
    "max_splats": 3000000,    # Max 3 million splats
    "max_resolution": 8192,   # Max resolution (limited by WebGPU dispatch group max of 65535)
    "shutdown_after_training": False # Shutdown PC after training
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

    async def _run_colmap_pipeline(self, colmap_dir: Path, images_dir: Path, input_type: str, colmap_cfg: dict, log_callback: Callable[[str], None]):
        """Run an explicit staged COLMAP pipeline (feature extraction, matching, mapping).

        Produces the sparse model under <colmap_dir>/sparse so the existing
        "largest model" discovery logic works unchanged. The database is kept at
        <colmap_dir>/database.db.
        """
        database_path = colmap_dir / "database.db"
        quality = colmap_cfg.get("quality", "high")
        engine = colmap_cfg.get("engine", "glomap")
        matcher = colmap_cfg.get("matcher", "auto")

        # Resolve effective matcher: auto -> sequential for video, exhaustive for images
        if matcher == "auto":
            effective_matcher = "sequential" if input_type == "video" else "exhaustive"
        else:
            effective_matcher = matcher

        # ---- Stage 1: Feature extraction ----
        await log_callback("Running feature extraction...\n")
        feature_args = [
            "feature_extractor",
            f"--database_path \"{database_path}\"",
            f"--image_path \"{images_dir}\"",
            "--FeatureExtraction.use_gpu 1",
        ]

        # Quality presets
        if quality == "low":
            feature_args.append("--FeatureExtraction.max_image_size 1600")
            feature_args.append("--SiftExtraction.max_num_features 4096")
        elif quality == "medium":
            feature_args.append("--FeatureExtraction.max_image_size 3200")
            feature_args.append("--SiftExtraction.max_num_features 8192")
        else:  # high
            feature_args.append("--FeatureExtraction.max_image_size 6400")
            feature_args.append("--SiftExtraction.max_num_features 16384")
            feature_args.append("--SiftExtraction.estimate_affine_shape 1")
            feature_args.append("--SiftExtraction.domain_size_pooling 1")

        # Video: one shared camera with a better distortion model for phones/action cams
        if input_type == "video":
            feature_args.append("--ImageReader.single_camera 1")
            feature_args.append("--ImageReader.camera_model OPENCV")

        await self.run_command(COLMAP_BAT_PATH, feature_args, log_callback)

        # ---- Stage 2: Matching ----
        guided = "--FeatureMatching.guided_matching 1" if quality == "high" else None

        if effective_matcher == "exhaustive":
            await log_callback("Running exhaustive matching...\n")
            match_args = [
                "exhaustive_matcher",
                f"--database_path \"{database_path}\"",
                "--FeatureMatching.use_gpu 1",
            ]
            if guided:
                match_args.append(guided)
            await self.run_command(COLMAP_BAT_PATH, match_args, log_callback)
        else:  # sequential
            await log_callback("Running sequential matching...\n")
            match_args = [
                "sequential_matcher",
                f"--database_path \"{database_path}\"",
                "--FeatureMatching.use_gpu 1",
                "--SequentialMatching.overlap 10",
            ]
            if guided:
                match_args.append(guided)
            await self.run_command(COLMAP_BAT_PATH, match_args, log_callback)

        # ---- Stage 3: Mapping ----
        sparse_out = colmap_dir / "sparse"
        ensure_directory(sparse_out)

        if engine == "glomap":
            # Best-effort focal calibration (important for EXIF-less video). If it
            # fails, log a warning and continue.
            await log_callback("Running view graph calibration (best effort)...\n")
            try:
                await self.run_command(COLMAP_BAT_PATH, [
                    "view_graph_calibrator",
                    f"--database_path \"{database_path}\"",
                ], log_callback)
            except Exception as e:
                logger.warning(f"view_graph_calibrator failed (continuing): {e}")
                await log_callback(f"WARNING: view_graph_calibrator failed, continuing: {e}\n")

            await log_callback("Running global mapper (GLOMAP)...\n")
            await self.run_command(COLMAP_BAT_PATH, [
                "global_mapper",
                f"--database_path \"{database_path}\"",
                f"--image_path \"{images_dir}\"",
                f"--output_path \"{sparse_out}\"",
            ], log_callback)
        else:  # incremental
            await log_callback("Running incremental mapper...\n")
            await self.run_command(COLMAP_BAT_PATH, [
                "mapper",
                f"--database_path \"{database_path}\"",
                f"--image_path \"{images_dir}\"",
                f"--output_path \"{sparse_out}\"",
            ], log_callback)

        # ---- Stage 4: Dense reconstruction (optional) ----
        if colmap_cfg.get("dense", 0):
            await log_callback("Running dense reconstruction...\n")
            # Find the largest sparse model (mirrors the discovery logic below)
            chosen_model = self._find_largest_sparse_model(sparse_out)
            if chosen_model is None:
                raise Exception("COLMAP did not produce a valid sparse model for dense reconstruction.")

            dense_dir = colmap_dir / "dense"
            await self.run_command(COLMAP_BAT_PATH, [
                "image_undistorter",
                f"--image_path \"{images_dir}\"",
                f"--input_path \"{chosen_model}\"",
                f"--output_path \"{dense_dir}\"",
                "--output_type COLMAP",
            ], log_callback)
            await self.run_command(COLMAP_BAT_PATH, [
                "patch_match_stereo",
                f"--workspace_path \"{dense_dir}\"",
            ], log_callback)
            await self.run_command(COLMAP_BAT_PATH, [
                "stereo_fusion",
                f"--workspace_path \"{dense_dir}\"",
                f"--output_path \"{dense_dir / 'fused.ply'}\"",
            ], log_callback)

    def _find_largest_sparse_model(self, colmap_sparse: Path) -> Optional[Path]:
        """Return the path of the largest sparse model under colmap_sparse, or None."""
        if not colmap_sparse.exists():
            return None
        sparse_candidates = []
        for folder in colmap_sparse.iterdir():
            if folder.is_dir():
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
                return colmap_sparse
            return None
        sparse_candidates.sort(key=lambda x: x[0], reverse=True)
        return sparse_candidates[0][1]

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

    def _build_lichtfeld_args(self, task_dir: Path, model_dir: Path, settings: dict) -> List[str]:
        """Build LichtFeld-Studio args."""
        args = [
            f"--data-path \"{task_dir}\"",
            f"--output-path \"{model_dir}\"",
            f"--iter {int(settings.get('total_steps', 30000))}",
            f"--sh-degree {int(settings.get('sh_degree', 3))}",
            f"--max-cap {int(settings.get('max_splats', 3000000))}",
            "--undistort"
        ]
        if not settings.get("with_viewer", True):
            args.append("--headless")
        else:
            args.append("--train")
        return args

    def _build_resume_lichtfeld_args(self, task_dir: Path, model_dir: Path, checkpoint_file: Path, settings: dict) -> List[str]:
        """Build LichtFeld-Studio args for resuming training."""
        args = [
            f"--resume \"{checkpoint_file}\"",
            f"--output-path \"{model_dir}\"",
            f"--iter {int(settings.get('total_steps', 30000))}",
            f"--sh-degree {int(settings.get('sh_degree', 3))}",
            f"--max-cap {int(settings.get('max_splats', 3000000))}",
            "--undistort"
        ]
        if not settings.get("with_viewer", True):
            args.append("--headless")
        else:
            args.append("--train")
        return args

    def _resolve_2dgs_r_value(self, images_dir: Path, max_resolution: int) -> int:
        """Translate Brush-style max_resolution (pixel cap) into a 2DGS -r value.

        2DGS treats -r values in {1, 2, 4, 8} as downscale factors; any other
        value is interpreted as an exact target image WIDTH. Passing the Brush
        cap directly (e.g. -r 8192) therefore UPSCALES captures to that width.
        Compute a factor that only ever downscales.
        """
        max_resolution = int(max_resolution)
        widths = []
        try:
            for f in sorted(Path(images_dir).iterdir()):
                if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                    continue
                img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    widths.append(img.shape[1])
                if len(widths) >= 8:
                    break
        except OSError as e:
            logger.warning(f"Could not probe images for 2DGS resolution: {e}")
        if not widths:
            return 1

        orig_w = int(np.median(widths))
        if orig_w <= max_resolution:
            return 1
        for factor in (2, 4, 8):
            if round(orig_w / factor) <= max_resolution:
                return factor
        return max_resolution

    def _build_2dgs_args(self, task_dir: Path, model_dir: Path, settings: dict) -> List[str]:
        """Build 2DGS (2D Gaussian Splatting) train.py args."""
        total_steps = int(settings.get('total_steps', 30000))
        # Checkpoint every 7000 steps (plus the final step) so training can be resumed
        ckpt_iters = [i for i in range(7000, total_steps, 7000)] + [total_steps]
        r_value = self._resolve_2dgs_r_value(Path(task_dir) / "images", int(settings.get('max_resolution', 8192)))
        args = [
            f"\"{TWO_DGS_DIR / 'train.py'}\"",
            f"-s \"{task_dir}\"",
            f"-m \"{model_dir}\"",
            f"--iterations {total_steps}",
            f"--sh_degree {int(settings.get('sh_degree', 3))}",
            f"-r {r_value}",
            "--checkpoint_iterations " + " ".join(str(i) for i in ckpt_iters),
            "--quiet"
        ]
        if any((Path(task_dir) / "lidar_depth").glob("*.npz")):
            args.append("--use_lidar_depth")
        return args

    def _build_resume_2dgs_args(self, task_dir: Path, model_dir: Path, checkpoint_file: Path, settings: dict) -> List[str]:
        """Build 2DGS train.py args for resuming from a chkpnt{iter}.pth file."""
        args = self._build_2dgs_args(task_dir, model_dir, settings)
        return args + [f"--start_checkpoint \"{checkpoint_file}\""]

    async def _export_2dgs_as_3dgs(self, model_dir: Path, brush_cfg: dict, log_callback: Callable[[str], None]):
        """Convert the newest 2DGS point_cloud.ply (flat disks, 2 scales) into a
        3DGS-compatible PLY at model/export_<iter>.ply so it can be opened in
        Brush, LichtFeld-Studio and standard splat viewers."""
        ply_candidates = sorted(model_dir.glob("point_cloud/iteration_*/point_cloud.ply"),
                                key=lambda p: int(p.parent.name.split("_")[-1]))
        if not ply_candidates:
            await log_callback("WARNING: no 2DGS point_cloud.ply found to export.\n")
            return

        src = ply_candidates[-1]
        iteration = src.parent.name.split("_")[-1]
        dst = model_dir / f"export_{iteration}.ply"

        script = TWO_DGS_DIR / "scripts" / "convert_2dgs_to_3dgs.py"
        args = [
            f"\"{script}\"",
            f"\"{src}\"",
            f"\"{dst}\"",
        ]
        await self.run_command(TWO_DGS_PYTHON, args, log_callback, cwd=TWO_DGS_DIR)
        await log_callback(f"Exported 3DGS-compatible splat for Brush/LichtFeld viewers: {dst.name}\n")

    async def _launch_training(self, task_dir: Path, model_dir: Path, brush_cfg: dict, log_callback: Callable[[str], None]):
        """Launch the configured trainer (Brush, LichtFeld-Studio or 2DGS) on a prepared project folder."""
        trainer = brush_cfg.get("trainer", "brush")

        if trainer == "lichtfeld":
            await log_callback("--- Running LichtFeld-Studio ---\n")

            async def lfs_monitor(proc, term_ctx):
                await self.monitor_lichtfeld_completion(proc, term_ctx, model_dir, int(brush_cfg.get('total_steps', 30000)), log_callback)

            monitor_enabled = brush_cfg.get("with_viewer") and brush_cfg.get("shutdown_after_training")
            await self.run_command(LICHTFELD_PATH, self._build_lichtfeld_args(task_dir, model_dir, brush_cfg), log_callback, cwd=task_dir, monitor_completion=lfs_monitor if monitor_enabled else None)
        elif trainer == "2dgs":
            await log_callback("--- Running 2DGS (2D Gaussian Splatting) ---\n")

            # Fresh start: remove leftovers from any previous/crashed run so
            # checkpoints and tensorboard logs don't mix across runs.
            if model_dir.exists():
                await log_callback("Removing existing model directory to start fresh...\n")
                shutil.rmtree(model_dir)

            # 2DGS always exposes a remote GUI socket; a SIBR viewer (e.g. GS_Monitor)
            # can connect to port 6009 while training runs.
            await log_callback("[INFO] 2DGS viewer socket available on port 6009 (connect with GS_Monitor/SIBR viewer).\n")

            await self.run_command(TWO_DGS_PYTHON, self._build_2dgs_args(task_dir, model_dir, brush_cfg), log_callback, cwd=TWO_DGS_DIR)
            await self._export_2dgs_as_3dgs(model_dir, brush_cfg, log_callback)
        else:
            await log_callback("--- Running Brush ---\n")

            async def brush_monitor(proc, term_ctx):
                await self.monitor_brush_completion(proc, term_ctx, model_dir, int(brush_cfg.get('total_steps', 30000)), log_callback)

            monitor_enabled = brush_cfg.get("with_viewer") and brush_cfg.get("shutdown_after_training")
            await self.run_command(BRUSH_PATH, self._build_brush_args(task_dir, model_dir, brush_cfg), log_callback, cwd=task_dir, monitor_completion=brush_monitor if monitor_enabled else None)

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
                    # Find available PLY/Resume checkpoints
                    ply_files = []
                    if model_path.exists():
                        # Brush PLY exports
                        for ply in model_path.glob("export_*.ply"):
                            try:
                                iter_str = ply.stem.replace("export_", "")
                                iteration = int(iter_str)
                                ply_files.append({
                                    "filename": ply.name,
                                    "iteration": iteration,
                                    "type": "brush"
                                })
                            except ValueError:
                                pass
                        
                        # LichtFeld resume files (e.g. checkpoints/checkpoint_*.resume or model/checkpoint_*.resume)
                        for resume in model_path.rglob("*.resume"):
                            try:
                                # typically checkpoint_30000.resume
                                iter_str = resume.stem.replace("checkpoint_", "")
                                iteration = int(iter_str)
                                ply_files.append({
                                    "filename": str(resume.relative_to(model_path)),
                                    "iteration": iteration,
                                    "type": "lichtfeld"
                                })
                            except ValueError:
                                pass

                        # 2DGS checkpoints (model/chkpnt{iter}.pth)
                        for chkpnt in model_path.glob("chkpnt*.pth"):
                            try:
                                iter_str = chkpnt.stem.replace("chkpnt", "")
                                iteration = int(iter_str)
                                ply_files.append({
                                    "filename": str(chkpnt.relative_to(model_path)),
                                    "iteration": iteration,
                                    "type": "2dgs"
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

        shutdown_needed = brush_settings.get("shutdown_after_training", False)

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
                trainer = brush_cfg.get("trainer", "brush")
                
                if trainer == "lichtfeld":
                    await log_callback("--- Running LichtFeld-Studio ---\n")
                    # Lichtfeld doesn't need monitor since --headless exits on completion
                    # but if with_viewer is true, we might need a monitor.
                    # For now, let's assume headless exits automatically, and viewer stays open.
                    args = self._build_lichtfeld_args(task_dir, model_dir, brush_cfg)
                    
                    async def lfs_monitor(proc, term_ctx):
                        await self.monitor_lichtfeld_completion(proc, term_ctx, model_dir, int(brush_cfg.get('total_steps', 30000)), log_callback)
                        
                    monitor_enabled = brush_cfg.get("with_viewer") and brush_cfg.get("shutdown_after_training")
                    await self.run_command(LICHTFELD_PATH, args, log_callback, cwd=task_dir, monitor_completion=lfs_monitor if monitor_enabled else None)
                elif trainer == "2dgs":
                    await log_callback("--- Running 2DGS (2D Gaussian Splatting) ---\n")
                    await log_callback("[INFO] 2DGS viewer socket available on port 6009 (connect with GS_Monitor/SIBR viewer).\n")

                    await self.run_command(TWO_DGS_PYTHON, self._build_2dgs_args(task_dir, model_dir, brush_cfg), log_callback, cwd=TWO_DGS_DIR)
                    await self._export_2dgs_as_3dgs(model_dir, brush_cfg, log_callback)
                else:
                    await log_callback("--- Running Brush ---\n")
                    # Define monitor for Brush
                    async def brush_monitor(proc, term_ctx):
                        await self.monitor_brush_completion(proc, term_ctx, model_dir, int(brush_cfg.get('total_steps', 30000)), log_callback)
                    
                    brush_args = self._build_brush_args(task_dir, model_dir, brush_cfg)
                    monitor_enabled = brush_cfg.get("with_viewer") and brush_cfg.get("shutdown_after_training")
                    await self.run_command(BRUSH_PATH, brush_args, log_callback, cwd=task_dir, monitor_completion=brush_monitor if monitor_enabled else None)
                
                await log_callback("--- Training Completed Successfully ---\n")
                
                if shutdown_needed:
                    await self.trigger_shutdown(log_callback)

            except Exception as e:
                logger.error(f"Training failed: {e}")
                await log_callback(f"\nCRITICAL ERROR: {str(e)}\n")
                raise e

        else:
            # RESUME logic
            try:
                brush_cfg = self._merge_settings(brush_settings, DEFAULT_BRUSH_SETTINGS)
                trainer = brush_cfg.get("trainer", "brush")
                
                # Check for Lichtfeld or Brush checkpoint
                if trainer == "lichtfeld":
                    # For lichtfeld, we might resume from `.resume` or `.ply`
                    expected_checkpoint = model_dir / f"checkpoint_{start_iter}.resume"
                    if not expected_checkpoint.exists():
                        # Try inside checkpoints folder
                        expected_checkpoint = model_dir / "checkpoints" / f"checkpoint_{start_iter}.resume"
                        if not expected_checkpoint.exists():
                            # Maybe we are resuming a Brush PLY in Lichtfeld?
                            expected_checkpoint = model_dir / f"export_{start_iter}.ply"
                    
                    if not expected_checkpoint.exists():
                        raise Exception(f"Checkpoint for iteration {start_iter} not found for LichtFeld-Studio")
                        
                    await log_callback(f"--- Resuming Training from {expected_checkpoint.name} ---\n")
                    await log_callback(f"Project: {task_dir}\n")
                    await log_callback(f"Target steps: {brush_settings.get('total_steps', 30000)}\n")

                    args = self._build_resume_lichtfeld_args(task_dir, model_dir, expected_checkpoint, brush_cfg)
                    
                    async def lfs_monitor(proc, term_ctx):
                        await self.monitor_lichtfeld_completion(proc, term_ctx, model_dir, int(brush_cfg.get('total_steps', 30000)), log_callback)

                    monitor_enabled = brush_cfg.get("with_viewer") and brush_cfg.get("shutdown_after_training")
                    await self.run_command(LICHTFELD_PATH, args, log_callback, cwd=task_dir, monitor_completion=lfs_monitor if monitor_enabled else None)
                elif trainer == "2dgs":
                    expected_checkpoint = model_dir / f"chkpnt{start_iter}.pth"
                    if not expected_checkpoint.exists():
                        available = [p.name for p in model_dir.glob("chkpnt*.pth")]
                        raise Exception(f"2DGS checkpoint for iteration {start_iter} not found. Available: {available or 'none (checkpoints are saved every 7000 steps)'}")

                    await log_callback(f"--- Resuming 2DGS Training from {expected_checkpoint.name} ---\n")
                    await log_callback(f"Project: {task_dir}\n")
                    await log_callback(f"Target steps: {brush_settings.get('total_steps', 30000)}\n")

                    await self.run_command(TWO_DGS_PYTHON, self._build_resume_2dgs_args(task_dir, model_dir, expected_checkpoint, brush_cfg), log_callback, cwd=TWO_DGS_DIR)
                    await self._export_2dgs_as_3dgs(model_dir, brush_cfg, log_callback)
                else:
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

                    # Define monitor for Brush
                    async def brush_monitor(proc, term_ctx):
                        await self.monitor_brush_completion(proc, term_ctx, model_dir, int(brush_cfg.get('total_steps', 30000)), log_callback)
                    
                    brush_args = self._build_resume_brush_args(task_dir, model_dir, start_iter, brush_cfg)
                    monitor_enabled = brush_cfg.get("with_viewer") and brush_cfg.get("shutdown_after_training")
                    await self.run_command(BRUSH_PATH, brush_args, log_callback, cwd=task_dir, monitor_completion=brush_monitor if monitor_enabled else None)
                
                await log_callback("--- Resume Training Completed Successfully ---\n")

                if shutdown_needed:
                    await self.trigger_shutdown(log_callback)
            except Exception as e:
                logger.error(f"Resume training failed: {e}")
                await log_callback(f"\nCRITICAL ERROR: {str(e)}\n")
                raise e

    async def _read_stream_lines(self, stream, callback: Optional[Callable[[str], None]]):
        """Read a subprocess stream splitting on newlines AND carriage returns.

        Progress bars (tqdm etc.) emit \\r-separated updates without ever
        sending \\n; readline() would accumulate them into one giant line until
        the StreamReader limit overflows ("Separator is not found, and chunk
        exceed the limit"). Splitting raw bytes also avoids UTF-8 chunk
        boundary issues.
        """
        pending = b""
        while True:
            chunk = await stream.read(65536)
            if not chunk:
                break
            parts = re.split(b"\r\n|\r|\n", pending + chunk)
            pending = parts.pop()
            for part in parts:
                await self._emit_cmd_output(part, callback)
        await self._emit_cmd_output(pending, callback)

    async def _emit_cmd_output(self, raw: bytes, callback: Optional[Callable[[str], None]]):
        try:
            decoded = raw.decode('utf-8').strip()
        except UnicodeDecodeError:
            decoded = raw.decode('cp1252', errors='replace').strip()
        if not decoded:
            return
        logger.info(f"CMD OUT: {decoded}")
        if callback:
            await callback(f"{decoded}\n")

    async def run_command(self, command: str, args: List[str], log_callback: Optional[Callable[[str], None]] = None, cwd: Optional[Path] = None, monitor_completion: Optional[Callable[[asyncio.subprocess.Process, dict], asyncio.Task]] = None):
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

        # Context to track if we intentionally killed the process
        termination_context = {"intentional": False}

        # Start monitor if provided
        monitor_task = None
        if monitor_completion:
            monitor_task = asyncio.create_task(monitor_completion(process, termination_context))

        try:
            await asyncio.gather(
                self._read_stream_lines(process.stdout, log_callback),
                self._read_stream_lines(process.stderr, log_callback)
            )
        except Exception as e:
            # A stream reader died unexpectedly. Don't leave an orphaned child
            # occupying the GPU and ports — kill the whole process tree.
            logger.error(f"Output stream reader failed: {e}. Terminating child process tree.")
            termination_context["intentional"] = True
            try:
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], capture_output=True)
            except Exception as kill_err:
                logger.error(f"Failed to terminate child process: {kill_err}")
            if log_callback:
                await log_callback(f"CRITICAL ERROR: output stream failed ({e}); child process terminated.\n")
            raise

        return_code = await process.wait()
        
        # Cancel monitor if it's still running
        if monitor_task and not monitor_task.done():
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        # On Windows, terminating a process often returns 1 or other non-zero codes.
        # If we intentionally killed it, we treat it as success.
        if return_code != 0 and not termination_context["intentional"]:
            error_msg = f"Command failed with exit code {return_code}"
            logger.error(error_msg)
            if log_callback:
                await log_callback(f"ERROR: {error_msg}\n")
            raise Exception(error_msg)
        
        logger.info(f"Command finished {'successfully' if return_code == 0 or termination_context['intentional'] else 'with code ' + str(return_code)}")
        if log_callback:
            await log_callback("Command finished successfully\n")

    async def monitor_brush_completion(self, process: asyncio.subprocess.Process, termination_context: dict, model_dir: Path, target_steps: int, log_callback: Callable[[str], None]):
        """Periodically checks for the final export PLY file and terminates the process if found."""
        target_file = model_dir / f"export_{target_steps:05d}.ply"
        # Also check without leading zeros just in case
        target_file_alt = model_dir / f"export_{target_steps}.ply"
        
        logger.info(f"Monitoring for completion file: {target_file}")
        
        while process.returncode is None:
            if target_file.exists() or target_file_alt.exists():
                await log_callback(f"\n[INFO] Final export detected. Training at target steps ({target_steps}) is complete.\n")
                await log_callback("[INFO] Automatically closing Brush to proceed...\n")
                logger.info("Completion file found. Terminating Brush process tree.")
                termination_context["intentional"] = True
                try:
                    # On Windows, we use taskkill to ensure the whole process tree (shell + app) is killed
                    if os.name == 'nt':
                        subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], capture_output=True)
                    else:
                        process.terminate()
                except Exception as e:
                    logger.error(f"Failed to terminate Brush: {e}")
                break
            await asyncio.sleep(5) # Poll every 5 seconds

    async def monitor_lichtfeld_completion(self, process: asyncio.subprocess.Process, termination_context: dict, model_dir: Path, target_steps: int, log_callback: Callable[[str], None]):
        """Periodically checks for the final export file and terminates the process if found."""
        target_file = model_dir / f"checkpoint_{target_steps}.resume"
        target_file_alt = model_dir / f"checkpoints/checkpoint_{target_steps}.resume"
        target_ply = model_dir / f"export_{target_steps}.ply"
        
        logger.info(f"Monitoring for completion file: {target_file}")
        
        while process.returncode is None:
            if target_file.exists() or target_file_alt.exists() or target_ply.exists():
                await log_callback(f"\n[INFO] Final export detected. Training at target steps ({target_steps}) is complete.\n")
                await log_callback("[INFO] Automatically closing LichtFeld-Studio to proceed...\n")
                logger.info("Completion file found. Terminating LichtFeld-Studio process tree.")
                termination_context["intentional"] = True
                try:
                    # On Windows, we use taskkill to ensure the whole process tree (shell + app) is killed
                    if os.name == 'nt':
                        subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)], capture_output=True)
                    else:
                        process.terminate()
                except Exception as e:
                    logger.error(f"Failed to terminate LichtFeld-Studio: {e}")
                break
            await asyncio.sleep(5) # Poll every 5 seconds

    async def trigger_shutdown(self, log_callback: Callable[[str], None]):
        """Triggers system shutdown after a delay."""
        logger.warning("Initiating system shutdown sequence...")
        await log_callback("\n[WARNING] System will shut down in 60 seconds...\n")
        
        # Windows shutdown command
        cmd = "shutdown /s /t 60"
        
        try:
            # We don't await this because we want to return and finish the request, 
            # letting the OS handle the shutdown timer.
            process = await asyncio.create_subprocess_shell(cmd)
            await log_callback("Shutdown command sent. Run 'shutdown /a' to abort.\n")
        except Exception as e:
            logger.error(f"Failed to trigger shutdown: {e}")
            await log_callback(f"Failed to trigger shutdown: {e}\n")

    async def run_sharp(self, input_path: str, output_path: str, device: str, render: bool, log_callback: Callable[[str], None]):
        """
        Run Sharp model for single-image 3DGS generation.
        
        Args:
            input_path: Path to the input image
            output_path: Path to save output Gaussians
            device: Device to run on (cuda, cpu, mps)
            render: Whether to render trajectory video
            log_callback: Callback for streaming logs
        """
        await log_callback("--- Running SHARP (Single Image 3DGS) ---\n")
        await log_callback(f"Input: {input_path}\n")
        await log_callback(f"Output: {output_path}\n")
        await log_callback(f"Device: {device}, Render: {render}\n")
        
        # Path to the pre-downloaded checkpoint (avoids SSL issues during download)
        import os
        checkpoint_path = Path(os.path.expanduser("~")) / ".cache" / "torch" / "hub" / "checkpoints" / "sharp_2572gikvuh.pt"
        
        # Build Sharp arguments
        args = [
            "predict",
            f"-i \"{input_path}\"",
            f"-o \"{output_path}\"",
            f"--device {device}",
        ]
        
        # Use pre-downloaded checkpoint if available
        if checkpoint_path.exists():
            args.append(f"-c \"{checkpoint_path}\"")
            await log_callback(f"Using cached checkpoint: {checkpoint_path}\n")
        
        if render:
            args.append("--render")
        else:
            args.append("--no-render")
        
        try:
            await self.run_command(SHARP_PATH, args, log_callback)
            await log_callback("--- SHARP Processing Completed ---\n")
        except Exception as e:
            logger.error(f"Sharp processing failed: {e}")
            await log_callback(f"\nCRITICAL ERROR: {str(e)}\n")
            raise e

    def calculate_md5(self, file_path: Path, chunk_size: int = 8192) -> str:
        """
        Calculate MD5 hash of a file.
        """
        md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                md5.update(chunk)
        return md5.hexdigest()

    def remove_duplicates(self, images_dir: Path, log_callback: Callable[[str], None]) -> int:
        """
        Find and remove duplicate images using MD5 hashing (exact duplicates).
        Returns number of removed images.
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        hashes = {}
        duplicates = []
        
        # Sort files to ensure deterministic behavior (e.g. keep the first one alphabetically)
        files = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions])
        
        count = 0
        total = len(files)
        
        logger.info(f"Scanning {total} files for duplicates using MD5...")
        
        for file_path in files:
            try:
                # Compute MD5
                file_hash = self.calculate_md5(file_path)
                
                if file_hash in hashes:
                    duplicates.append(file_path)
                else:
                    hashes[file_hash] = file_path
            except Exception as e:
                logger.error(f"Error checking duplicate for {file_path}: {e}")
        
        # Remove duplicates
        removed_count = 0
        for dup in duplicates:
            try:
                dup.unlink()
                removed_count += 1
                logger.info(f"Removed duplicate image: {dup.name}")
            except Exception as e:
                logger.error(f"Failed to delete duplicate {dup.name}: {e}")
                
        return removed_count

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
        
        ensure_directory(output_dir)

        # Determine candidate frames (per existing step logic)
        candidate_indices = []
        count = 0
        next_frame_to_save = 0.0
        while True:
            if count >= next_frame_to_save:
                candidate_indices.append(count)
                next_frame_to_save += step
            count += 1
            if count >= total_frames:
                break

        blur_filter = settings.get("blur_filter", True)
        use_blur_filter = blur_filter and len(candidate_indices) >= 10

        if use_blur_filter:
            # Pass 1: decode video, compute Laplacian variance score for each candidate
            scores = []
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise Exception("Failed to open video file")
            idx = 0
            cand_set = set(candidate_indices)
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx in cand_set:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    score = cv2.Laplacian(gray, cv2.CV_64F).var()
                    scores.append(score)
                frame_idx += 1
            cap.release()

            # Determine threshold: keep frames with score >= 0.25 * median(scores)
            import statistics
            if scores:
                median_score = statistics.median(scores)
                threshold = 0.25 * median_score
                keep_mask = [s >= threshold for s in scores]

                # If that would keep fewer than half, keep the top 50% by score instead
                if sum(keep_mask) < len(scores) / 2:
                    keep_mask = [False] * len(scores)
                    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                    keep_count = max(1, len(scores) // 2)
                    for i in sorted_idx[:keep_count]:
                        keep_mask[i] = True

                dropped = len(scores) - sum(keep_mask)
                logger.info(f"Blur filter: dropped {dropped} of {len(scores)} candidate frames (threshold={threshold:.2f})")
            else:
                keep_mask = []
                dropped = 0
                threshold = 0.0

            # Pass 2: decode again, write only kept candidates with contiguous zero-padded names
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise Exception("Failed to open video file")
            saved_count = 0
            frame_idx = 0
            score_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx in cand_set:
                    if keep_mask[score_idx]:
                        frame_name = f"frame_{saved_count:05d}.jpg"
                        cv2.imwrite(str(output_dir / frame_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                        saved_count += 1
                    score_idx += 1
                frame_idx += 1
            cap.release()
            blur_info = {"dropped": dropped, "total": len(scores), "threshold": threshold}
            return saved_count, orig_fps, blur_info

        # Single-pass behavior (blur filter disabled or < 10 candidates)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise Exception("Failed to open video file")

        saved_count = 0
        next_frame_to_save = 0.0
        count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if count >= next_frame_to_save:
                frame_name = f"frame_{saved_count:05d}.jpg"
                cv2.imwrite(str(output_dir / frame_name), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                saved_count += 1
                next_frame_to_save += step
            
            count += 1

        cap.release()
        return saved_count, orig_fps, None

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
                blur_active = extraction_settings.get("blur_filter", True)
                await log_callback(f"Extracting frames from video... Mode: {extraction_settings.get('mode')} Value: {extraction_settings.get('value')} Blur filter: {'on' if blur_active else 'off'}\n")
                
                # Run extraction in thread pool to avoid blocking asyncio loop
                loop = asyncio.get_running_loop()
                num_extracted, orig_fps, blur_info = await loop.run_in_executor(
                    None, 
                    self.extract_frames, 
                    input_path, 
                    images_dir, 
                    extraction_settings, 
                    log_callback
                )
                
                await log_callback(f"Extracted {num_extracted} frames (Source: {orig_fps:.2f} fps).\n")
                if blur_info:
                    await log_callback(f"Blur filter: dropped {blur_info['dropped']} of {blur_info['total']} candidate frames (threshold={blur_info['threshold']:.2f})\n")
                
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

            # 1.5 Duplicate Removal (if enabled)
            if colmap_cfg.get("remove_duplicates", False):
                await log_callback("Checking for duplicate images...\n")
                loop = asyncio.get_running_loop()
                removed = await loop.run_in_executor(None, self.remove_duplicates, images_dir, log_callback)
                if removed > 0:
                    await log_callback(f"Removed {removed} duplicate images.\n")
                else:
                    await log_callback("No duplicates found.\n")

            # 2. COLMAP
            await log_callback("--- Step 2: Running COLMAP (Tracking) ---\n")
            
            # Check if we have enough images
            valid_images = list(images_dir.glob("*.tif")) + list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            if len(valid_images) < 2:
                msg = f"Insufficient images for COLMAP reconstruction. Found {len(valid_images)}. Need at least 2 (preferably 20+)."
                await log_callback(f"CRITICAL ERROR: {msg}\n")
                raise Exception(msg)

            # We use an explicit staged COLMAP pipeline (feature extraction,
            # matching, mapping) instead of 'automatic_reconstructor'.
            database_path = colmap_dir / "database.db"

            await self._run_colmap_pipeline(colmap_dir, images_dir, input_type, colmap_cfg, log_callback)

            # 3. Training
            trainer_labels = {
                "lichtfeld": "LichtFeld-Studio",
                "2dgs": "2DGS (2D Gaussian Splatting)",
                "brush": "Brush"
            }
            await log_callback(f"--- Step 3: Running {trainer_labels.get(brush_cfg.get('trainer', 'brush'), 'Brush')} (Training) ---\n")

            # Both trainers need the colmap sparse output at <task>/sparse/0.
            # The pipeline produces the sparse model under colmap_dir/sparse.
            final_sparse_dir = task_dir / "sparse"
            if final_sparse_dir.exists():
                shutil.rmtree(final_sparse_dir)
            ensure_directory(final_sparse_dir)

            # Find the largest sparse model in colmap_dir/sparse
            colmap_sparse = colmap_dir / "sparse"
            best_model_path = self._find_largest_sparse_model(colmap_sparse)

            if best_model_path is None:
                raise Exception("COLMAP did not produce a valid sparse model (points3D not found).")

            if best_model_path == colmap_sparse:
                # Flat structure: the sparse folder IS the model itself
                shutil.copytree(colmap_sparse, final_sparse_dir / "0")
                logger.info("Copied flat sparse model to sparse/0")
            else:
                logger.info(f"Multiple COLMAP models found. Selected largest: {best_model_path.name}")
                shutil.copytree(best_model_path, final_sparse_dir / "0")

            await self._launch_training(task_dir, model_dir, brush_cfg, log_callback)

            await log_callback("--- Pipeline Completed Successfully ---\n")

            if brush_cfg.get("shutdown_after_training"):
                await self.trigger_shutdown(log_callback)

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            await log_callback(f"\nCRITICAL ERROR: {str(e)}\n")
            raise e

    # ------------------------------------------------------------------
    # LiDAR capture (SplatKing-style ZIP) workflow
    # ------------------------------------------------------------------

    def _extract_zip(self, zip_path: Path, dest_dir: Path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)

    def _find_lidar_model_root(self, extract_dir: Path) -> Optional[Path]:
        """Locate a COLMAP text model (sparse/0/cameras.txt) inside an extracted capture."""
        for candidate in extract_dir.rglob("cameras.txt"):
            if candidate.parent.name == "0" and candidate.parent.parent.name == "sparse":
                return candidate.parent.parent.parent
        return None

    def _parse_first_camera_params(self, cameras_txt: Path) -> Optional[str]:
        """Extract 'MODEL WIDTH HEIGHT PARAMS[]' from cameras.txt, returning PARAMS as csv."""
        try:
            with open(cameras_txt, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) >= 8:
                        return ",".join(parts[4:])
        except Exception as e:
            logger.error(f"Failed to parse {cameras_txt}: {e}")
        return None

    async def _convert_colmap_txt_to_bin(self, txt_model_dir: Path, out_dir: Path, log_callback: Callable[[str], None]):
        """Convert a COLMAP text model to binary format (required by Brush/LichtFeld)."""
        ensure_directory(out_dir)
        await self.run_command(COLMAP_BAT_PATH, [
            "model_converter",
            f"--input_path \"{txt_model_dir}\"",
            f"--output_path \"{out_dir}\"",
            "--output_type BIN",
        ], log_callback)

        missing = [n for n in ("cameras.bin", "images.bin", "points3D.bin") if not (out_dir / n).exists()]
        if missing:
            raise Exception(f"model_converter did not produce: {', '.join(missing)}")

    async def _run_colmap_direct(self, args: List[str], log_callback: Callable[[str], None]):
        """Run colmap.exe directly (bypasses COLMAP.bat, whose %* handling breaks
        arguments containing quoted comma-separated values like camera_params)."""
        colmap_root = Path(COLMAP_EXE_PATH).parent.parent
        env = os.environ.copy()
        env["PATH"] = f"{colmap_root / 'bin'};{env.get('PATH', '')}"
        env["QT_PLUGIN_PATH"] = f"{colmap_root / 'plugins'};{env.get('QT_PLUGIN_PATH', '')}"

        full_command = f'"{COLMAP_EXE_PATH}" ' + " ".join(args)
        logger.info(f"Starting command: {full_command}")
        if log_callback:
            await log_callback(f"Executing: {full_command}\n")

        process = await asyncio.create_subprocess_shell(
            full_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env
        )

        await asyncio.gather(
            self._read_stream_lines(process.stdout, log_callback),
            self._read_stream_lines(process.stderr, log_callback)
        )

        return_code = await process.wait()
        if return_code != 0:
            error_msg = f"Command failed with exit code {return_code}"
            logger.error(error_msg)
            if log_callback:
                await log_callback(f"ERROR: {error_msg}\n")
            raise Exception(error_msg)

        if log_callback:
            await log_callback("Command finished successfully\n")

    async def _refine_lidar_poses(self, colmap_dir: Path, images_dir: Path, txt_model_dir: Path, refined_out: Path, camera_params: str, log_callback: Callable[[str], None]):
        """Refine device-provided poses with real image observations.

        Runs SIFT extraction + sequential matching, then point_triangulator which
        keeps the device poses/points as prior, adds 2D-3D tracks from matches
        and bundle-adjusts poses anchored to that prior.
        """
        database_path = colmap_dir / "database.db"

        await log_callback("Running feature extraction on capture images...\n")
        await self._run_colmap_direct([
            "feature_extractor",
            f"--database_path \"{database_path}\"",
            f"--image_path \"{images_dir}\"",
            "--ImageReader.single_camera 1",
            "--ImageReader.camera_model PINHOLE",
            f"--ImageReader.camera_params \"{camera_params}\"",
            "--FeatureExtraction.use_gpu 1",
            "--SiftExtraction.max_num_features 16384",
        ], log_callback)

        await log_callback("Running sequential matching...\n")
        await self._run_colmap_direct([
            "sequential_matcher",
            f"--database_path \"{database_path}\"",
            "--FeatureMatching.use_gpu 1",
            "--SequentialMatching.overlap 10",
            "--SequentialMatching.quadratic_overlap 1",
        ], log_callback)

        ensure_directory(refined_out)
        await log_callback("Running point triangulation + bundle adjustment (poses anchored to device tracking)...\n")
        await self._run_colmap_direct([
            "point_triangulator",
            f"--database_path \"{database_path}\"",
            f"--image_path \"{images_dir}\"",
            f"--input_path \"{txt_model_dir}\"",
            f"--output_path \"{refined_out}\"",
        ], log_callback)

        missing = [n for n in ("cameras.bin", "images.bin", "points3D.bin") if not (refined_out / n).exists()]
        if missing:
            raise Exception(f"point_triangulator did not produce: {', '.join(missing)}")

    async def _generate_lidar_depth_maps(self, task_dir: Path, brush_cfg: dict, log_callback: Callable[[str], None]):
        """Project the fused LiDAR cloud into every view to create depth
        supervision maps consumed by 2DGS train.py --use_lidar_depth."""
        sparse_dir = task_dir / "sparse" / "0"
        points_bin = task_dir / "colmap_workspace" / "converted_bin" / "points3D.bin"
        images_dir = task_dir / "images"
        output_dir = task_dir / "lidar_depth"

        if not points_bin.exists():
            await log_callback("WARNING: fused LiDAR cloud not found; skipping depth supervision.\n")
            return

        r_value = self._resolve_2dgs_r_value(images_dir, int(brush_cfg.get("max_resolution", 8192)))
        script = TWO_DGS_DIR / "scripts" / "gen_lidar_depth.py"
        args = [
            f"\"{script}\"",
            f"--sparse_dir \"{sparse_dir}\"",
            f"--points_bin \"{points_bin}\"",
            f"--images_dir \"{images_dir}\"",
            f"--output_dir \"{output_dir}\"",
            f"--train_res {r_value}",
        ]
        await self.run_command(TWO_DGS_PYTHON, args, log_callback, cwd=TWO_DGS_DIR)
        count = len(list(output_dir.glob("*.npz")))
        await log_callback(f"Generated {count} LiDAR depth map(s).\n")

    async def process_lidar_zip(self, task_id: str, zip_path: Path, log_callback: Callable[[str], None], colmap_settings: Optional[dict] = None, brush_settings: Optional[dict] = None, project_name: Optional[str] = None):
        """Pipeline for LiDAR capture ZIPs (e.g. SplatKing exports).

        These captures ship a COLMAP-compatible model whose poses come from
        device tracking (ARKit) and whose points come from the fused LiDAR cloud.
        Steps:
          1. Extract archive, stage images.
          2. Convert the shipped text model to binary; optionally refine poses
             via feature matching + triangulation/bundle adjustment.
          3. Train a Gaussian Splat (Brush or LichtFeld-Studio).
        """
        colmap_cfg = self._merge_settings(colmap_settings, DEFAULT_COLMAP_SETTINGS)
        brush_cfg = self._merge_settings(brush_settings, DEFAULT_BRUSH_SETTINGS)

        if project_name and project_name.strip():
            sanitized_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', project_name.strip())
            if not sanitized_name:
                sanitized_name = task_id

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

        source_dir = task_dir / "source"
        images_dir = task_dir / "images"
        model_dir = task_dir / "model"

        refine_poses = bool(colmap_cfg.get("refine_poses", True))

        try:
            # ---- Step 1: Extract & stage ----
            await log_callback("--- Step 1: Extracting LiDAR Capture ---\n")
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._extract_zip, zip_path, source_dir)

            model_root = self._find_lidar_model_root(source_dir)
            if model_root is None:
                raise Exception("No COLMAP_Text_Model (sparse/0/cameras.txt) found inside the ZIP.")
            await log_callback(f"Found COLMAP model: {model_root.name}\n")

            txt_model_dir = model_root / "sparse" / "0"
            src_images_dir = model_root / "images"
            if not src_images_dir.exists():
                raise Exception(f"No images folder inside the model: {src_images_dir}")

            image_count = len([f for f in src_images_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff")])
            if image_count < 2:
                raise Exception(f"Insufficient images in capture ({image_count}). Need at least 2.")
            await log_callback(f"Staging {image_count} images...\n")
            await loop.run_in_executor(None, shutil.copytree, src_images_dir, images_dir)

            # ---- Step 2: Prepare sparse model ----
            await log_callback("--- Step 2: Preparing Camera Tracking Data ---\n")
            workspace_dir = task_dir / "colmap_workspace"
            ensure_directory(workspace_dir)

            converted_bin = workspace_dir / "converted_bin"
            camera_params = self._parse_first_camera_params(txt_model_dir / "cameras.txt")
            await log_callback("Converting shipped text model to binary...\n")
            await self._convert_colmap_txt_to_bin(txt_model_dir, converted_bin, log_callback)

            final_sparse = task_dir / "sparse" / "0"
            if final_sparse.parent.exists():
                shutil.rmtree(final_sparse.parent)

            if refine_poses:
                await log_callback("Refining device poses with image observations...\n")
                refined_out = workspace_dir / "refined"
                try:
                    await self._refine_lidar_poses(
                        workspace_dir, images_dir, txt_model_dir, refined_out,
                        camera_params or "1,1,0.5,0.5",
                        log_callback,
                    )
                    shutil.copytree(refined_out, final_sparse)
                    await log_callback("Pose refinement succeeded - using refined model.\n")
                except Exception as e:
                    logger.warning(f"Pose refinement failed, falling back to shipped model: {e}")
                    await log_callback(f"WARNING: Pose refinement failed ({e}). Falling back to shipped device-pose model.\n")
                    shutil.rmtree(final_sparse)
                    ensure_directory(final_sparse)
                    shutil.copytree(converted_bin, final_sparse)
            else:
                shutil.copytree(converted_bin, final_sparse)
                await log_callback("Using shipped device-pose model (refinement disabled).\n")

            # ---- Step 2.5: LiDAR depth supervision maps (2DGS only) ----
            if brush_cfg.get("trainer", "brush") == "2dgs":
                await log_callback("--- Step 2.5: Generating LiDAR Depth Supervision Maps ---\n")
                try:
                    await self._generate_lidar_depth_maps(task_dir, brush_cfg, log_callback)
                except Exception as e:
                    logger.warning(f"LiDAR depth map generation failed: {e}")
                    await log_callback(f"WARNING: LiDAR depth map generation failed ({e}). Continuing without depth supervision.\n")

            # ---- Step 3: Training ----
            trainer_labels = {
                "lichtfeld": "LichtFeld-Studio",
                "2dgs": "2DGS (2D Gaussian Splatting)",
                "brush": "Brush"
            }
            await log_callback(f"--- Step 3: Running {trainer_labels.get(brush_cfg.get('trainer', 'brush'), 'Brush')} (Training) ---\n")

            ensure_directory(model_dir)
            await self._launch_training(task_dir, model_dir, brush_cfg, log_callback)

            await log_callback("--- Pipeline Completed Successfully ---\n")

            if brush_cfg.get("shutdown_after_training"):
                await self.trigger_shutdown(log_callback)

        except Exception as e:
            logger.error(f"LiDAR pipeline failed: {e}")
            await log_callback(f"\nCRITICAL ERROR: {str(e)}\n")
            raise e
