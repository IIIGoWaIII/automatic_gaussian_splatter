"""
Update Manager for Automatic Gaussian Splatter

Handles checking for updates and installing updates from GitHub for:
- COLMAP (pre-built Windows binaries)
- Brush (pre-built Windows binaries)
- Main App (git pull)
"""

import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import urllib.request
import urllib.error

from utils import logger, get_project_root

# GitHub API endpoints
GITHUB_API = "https://api.github.com"
COLMAP_REPO = "colmap/colmap"
BRUSH_REPO = "ArthurBrussee/brush"
APP_REPO = "IIIGoWaIII/automatic_gaussian_splatter"

# Asset names for Windows downloads
COLMAP_ASSET_NAME = "colmap-x64-windows-cuda.zip"
BRUSH_ASSET_NAME = "brush-app-x86_64-pc-windows-msvc.zip"

# Local folder names
COLMAP_FOLDER = "colmap-x64-windows-cuda"
BRUSH_FOLDER = "brush-app-x86_64-pc-windows-msvc"
SHARP_FOLDER = "ml-sharp"

# ML-Sharp specific
SHARP_REPO = "apple/ml-sharp"
SHARP_REPO_URL = "https://github.com/apple/ml-sharp.git"
SHARP_CHECKPOINT_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
SHARP_CHECKPOINT_NAME = "sharp_2572gikvuh.pt"


class UpdateManager:
    """Manages checking and installing updates for all components."""
    
    def __init__(self):
        # get_project_root() returns python_webui folder, but COLMAP/Brush are in the parent
        self.project_root = get_project_root().parent
        self.versions_file = self.project_root / "versions.json"
        self._ensure_versions_file()
    
    def _ensure_versions_file(self):
        """Create versions.json if it doesn't exist."""
        if not self.versions_file.exists():
            # Try to detect current versions
            versions = {
                "colmap": self._detect_colmap_version(),
                "brush": self._detect_brush_version(),
                "app_commit": self._get_local_git_commit()
            }
            self._save_versions(versions)
    
    def _detect_colmap_version(self) -> str:
        """Try to detect installed COLMAP version."""
        # Default to empty if we can't detect
        return ""
    
    def _detect_brush_version(self) -> str:
        """Try to detect installed Brush version."""
        # Check CHANGELOG.md for version info
        changelog = self.project_root / BRUSH_FOLDER / "CHANGELOG.md"
        if changelog.exists():
            try:
                content = changelog.read_text(encoding='utf-8')
                # Look for version pattern like "## 0.3.0" or "## v0.3.0"
                import re
                match = re.search(r'##\s*v?(\d+\.\d+\.\d+)', content)
                if match:
                    return f"v{match.group(1)}" if not match.group(0).startswith('## v') else match.group(1)
            except Exception:
                pass
        return ""
    
    def _get_local_git_commit(self) -> str:
        """Get the current git commit SHA of the main app."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.warning(f"Could not get git commit: {e}")
        return ""
    
    def _load_versions(self) -> Dict[str, str]:
        """Load versions from versions.json."""
        try:
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        except Exception:
            return {"colmap": "", "brush": "", "app_commit": ""}
    
    def _save_versions(self, versions: Dict[str, str]):
        """Save versions to versions.json."""
        try:
            with open(self.versions_file, 'w') as f:
                json.dump(versions, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save versions: {e}")
    
    def _github_api_request(self, endpoint: str) -> Optional[dict]:
        """Make a request to GitHub API."""
        url = f"{GITHUB_API}{endpoint}"
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'AutomaticGaussianSplatter/1.0')
            req.add_header('Accept', 'application/vnd.github.v3+json')
            
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            logger.warning(f"GitHub API error for {endpoint}: {e.code}")
            return None
        except Exception as e:
            logger.warning(f"GitHub API request failed: {e}")
            return None
    
    def _get_latest_colmap_release(self) -> Optional[Tuple[str, str]]:
        """Get latest COLMAP release tag and download URL."""
        data = self._github_api_request(f"/repos/{COLMAP_REPO}/releases/latest")
        if not data:
            return None
        
        tag = data.get("tag_name", "")
        assets = data.get("assets", [])
        
        for asset in assets:
            if asset.get("name") == COLMAP_ASSET_NAME:
                return (tag, asset.get("browser_download_url", ""))
        
        return (tag, "") if tag else None
    
    def _get_latest_brush_release(self) -> Optional[Tuple[str, str]]:
        """Get latest Brush release tag and download URL."""
        # Brush uses tags, try to get latest release first
        data = self._github_api_request(f"/repos/{BRUSH_REPO}/releases/latest")
        if data:
            tag = data.get("tag_name", "")
            assets = data.get("assets", [])
            
            for asset in assets:
                if asset.get("name") == BRUSH_ASSET_NAME:
                    return (tag, asset.get("browser_download_url", ""))
            
            # If no matching asset in latest release, check all releases
            if tag:
                return (tag, f"https://github.com/{BRUSH_REPO}/releases/download/{tag}/{BRUSH_ASSET_NAME}")
        
        # Fallback to tags
        tags_data = self._github_api_request(f"/repos/{BRUSH_REPO}/tags")
        if tags_data and len(tags_data) > 0:
            tag = tags_data[0].get("name", "")
            if tag:
                return (tag, f"https://github.com/{BRUSH_REPO}/releases/download/{tag}/{BRUSH_ASSET_NAME}")
        
        return None
    
    def _get_latest_app_commit(self) -> Optional[str]:
        """Get latest commit SHA from main app repo."""
        data = self._github_api_request(f"/repos/{APP_REPO}/commits?per_page=1")
        if data and len(data) > 0:
            return data[0].get("sha", "")
        return None
    
    def _get_latest_sharp_commit(self) -> Optional[str]:
        """Get latest commit SHA from ML-Sharp repo."""
        data = self._github_api_request(f"/repos/{SHARP_REPO}/commits?per_page=1")
        if data and len(data) > 0:
            return data[0].get("sha", "")
        return None
    
    def _get_local_sharp_commit(self) -> str:
        """Get the current git commit SHA of the local ml-sharp folder."""
        sharp_path = self.project_root / SHARP_FOLDER
        if not sharp_path.exists():
            return ""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(sharp_path),
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.warning(f"Could not get sharp git commit: {e}")
        return ""
    
    def _is_sharp_installed(self) -> bool:
        """Check if ML-Sharp is properly installed with venv."""
        sharp_path = self.project_root / SHARP_FOLDER
        venv_path = sharp_path / ".venv"
        sharp_exe = venv_path / "Scripts" / "sharp.exe"
        return sharp_exe.exists()
    
    def check_for_updates(self) -> Dict:
        """
        Check for available updates for all components.
        
        Returns:
            Dict with 'updates_available' bool and 'updates' list
        """
        local_versions = self._load_versions()
        updates = []
        
        # Check COLMAP
        colmap_info = self._get_latest_colmap_release()
        if colmap_info:
            latest_tag, download_url = colmap_info
            local_version = local_versions.get("colmap", "")
            if latest_tag and latest_tag != local_version:
                updates.append({
                    "name": "COLMAP",
                    "key": "colmap",
                    "current": local_version or "Unknown",
                    "latest": latest_tag,
                    "download_url": download_url
                })
        
        # Check Brush
        brush_info = self._get_latest_brush_release()
        if brush_info:
            latest_tag, download_url = brush_info
            local_version = local_versions.get("brush", "")
            if latest_tag and latest_tag != local_version:
                updates.append({
                    "name": "Brush",
                    "key": "brush",
                    "current": local_version or "Unknown",
                    "latest": latest_tag,
                    "download_url": download_url
                })
        
        # Check Main App
        latest_commit = self._get_latest_app_commit()
        if latest_commit:
            local_commit = local_versions.get("app_commit", "")
            if latest_commit and latest_commit != local_commit:
                updates.append({
                    "name": "Gaussian Splatter App",
                    "key": "app",
                    "current": local_commit[:8] if local_commit else "Unknown",
                    "latest": latest_commit[:8],
                    "download_url": ""  # Uses git pull
                })
        
        # Check ML-Sharp
        latest_sharp = self._get_latest_sharp_commit()
        if latest_sharp:
            local_sharp = self._get_local_sharp_commit()
            is_installed = self._is_sharp_installed()
            
            # Show update if: not installed OR commit differs
            if not is_installed or (latest_sharp and latest_sharp != local_sharp):
                updates.append({
                    "name": "ML-Sharp (Single Image 3DGS)",
                    "key": "sharp",
                    "current": local_sharp[:8] if local_sharp else ("Not Installed" if not is_installed else "Unknown"),
                    "latest": latest_sharp[:8],
                    "download_url": SHARP_REPO_URL
                })
        
        return {
            "updates_available": len(updates) > 0,
            "updates": updates
        }
    
    def _download_file(self, url: str, dest_path: Path, progress_callback=None) -> bool:
        """Download a file from URL to destination path."""
        try:
            req = urllib.request.Request(url)
            req.add_header('User-Agent', 'AutomaticGaussianSplatter/1.0')
            
            with urllib.request.urlopen(req, timeout=300) as response:
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                chunk_size = 8192
                
                with open(dest_path, 'wb') as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size:
                            progress_callback(downloaded, total_size)
            
            return True
        except Exception as e:
            logger.error(f"Download failed: {e}")
            return False
    
    def _safe_rmtree(self, path: Path, retries: int = 3, delay: float = 1.0) -> bool:
        """Safely remove a directory tree with retries for locked files."""
        import time
        import gc
        
        for attempt in range(retries):
            try:
                # Force garbage collection to release any Python references
                gc.collect()
                
                if path.exists():
                    shutil.rmtree(str(path), ignore_errors=False)
                return True
            except PermissionError as e:
                if attempt < retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{retries} - File locked: {e}")
                    time.sleep(delay)
                else:
                    # On final failure, try ignore_errors mode
                    logger.warning(f"Could not fully remove {path}, some files may be locked")
                    try:
                        shutil.rmtree(str(path), ignore_errors=True)
                    except:
                        pass
                    return False
            except Exception as e:
                logger.error(f"Error removing {path}: {e}")
                return False
        return True
    
    def _extract_and_replace(self, zip_path: Path, target_folder: str) -> bool:
        """Extract zip and replace existing folder."""
        import time
        
        target_path = self.project_root / target_folder
        backup_path = self.project_root / f"{target_folder}_backup"
        old_backup_path = self.project_root / f"{target_folder}_old"
        
        try:
            # Create temp extraction directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Extract zip
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_path)
                
                # Find the extracted folder (might be nested)
                extracted_items = list(temp_path.iterdir())
                if len(extracted_items) == 1 and extracted_items[0].is_dir():
                    source_path = extracted_items[0]
                else:
                    source_path = temp_path
                
                # Clean up any old backup first
                if old_backup_path.exists():
                    self._safe_rmtree(old_backup_path)
                
                # If there's a leftover backup from a failed previous attempt, move it
                if backup_path.exists():
                    try:
                        shutil.move(str(backup_path), str(old_backup_path))
                    except:
                        self._safe_rmtree(backup_path)
                
                # Backup existing folder by renaming (faster than copy)
                if target_path.exists():
                    shutil.move(str(target_path), str(backup_path))
                
                # Move new folder into place
                shutil.copytree(str(source_path), str(target_path))
                
                # Try to remove backup in background - don't fail if it's locked
                # The backup will be cleaned up on next update or can be manually deleted
                if backup_path.exists():
                    if not self._safe_rmtree(backup_path, retries=2, delay=0.5):
                        logger.info(f"Note: Old backup at {backup_path} could not be removed (files in use). It will be cleaned up later.")
                
            return True
        except Exception as e:
            logger.error(f"Extract and replace failed: {e}")
            # Try to restore backup
            if backup_path.exists() and not target_path.exists():
                try:
                    shutil.move(str(backup_path), str(target_path))
                except Exception as restore_err:
                    logger.error(f"Failed to restore backup: {restore_err}")
            return False
    
    def install_update(self, app_key: str, download_url: str, latest_version: str, 
                       log_callback=None) -> bool:
        """
        Install an update for the specified app.
        
        Args:
            app_key: 'colmap', 'brush', or 'app'
            download_url: URL to download from (empty for app which uses git)
            latest_version: The version string to save after update
            log_callback: Optional callback for progress messages
        
        Returns:
            True if update succeeded, False otherwise
        """
        def log(msg):
            if log_callback:
                log_callback(msg)
            logger.info(msg)
        
        if app_key == "app":
            return self._update_app(latest_version, log)
        elif app_key == "colmap":
            return self._update_component(
                "COLMAP", download_url, COLMAP_FOLDER, "colmap", latest_version, log
            )
        elif app_key == "brush":
            return self._update_component(
                "Brush", download_url, BRUSH_FOLDER, "brush", latest_version, log
            )
        elif app_key == "sharp":
            return self._install_sharp(latest_version, log)
        else:
            log(f"Unknown app key: {app_key}")
            return False

    def _get_python_executable(self, log) -> Optional[str]:
        """Find a suitable Python 3.13 executable."""
        # Try common names
        for cmd in ["python3.13", "python313", "python", "py"]:
            try:
                args = [cmd, "--version"]
                if cmd == "py":
                    args = ["py", "-3.13", "--version"]
                
                result = subprocess.run(args, capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version = result.stdout.strip() or result.stderr.strip()
                    if "3.13" in version:
                        return cmd if cmd != "py" else "py -3.13"
                    elif cmd == "python" or cmd == "python3":
                        # If it's just 'python', check if it's 3.13
                        if "3.13" in version:
                           return cmd
            except:
                continue
        return None

    def _install_sharp(self, latest_commit: str, log) -> bool:
        """Full installation flow for ML-Sharp."""
        log("Starting ML-Sharp installation/update...")
        
        # 1. Check Python 3.13
        python_exe = self._get_python_executable(log)
        if not python_exe:
            log("Error: Python 3.13 not found. ML-Sharp requires Python 3.13.")
            log("Please install Python 3.13 from python.org and try again.")
            return False
        
        sharp_path = self.project_root / SHARP_FOLDER
        
        try:
            # 2. Git Clone or Pull
            if not sharp_path.exists() or not (sharp_path / ".git").exists():
                if sharp_path.exists():
                    log("Cleaning up existing incomplete installation...")
                    self._safe_rmtree(sharp_path)
                
                log(f"Cloning ML-Sharp from {SHARP_REPO_URL}...")
                result = subprocess.run(
                    ["git", "clone", SHARP_REPO_URL, SHARP_FOLDER],
                    cwd=str(self.project_root),
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode != 0:
                    log(f"Clone failed: {result.stderr}")
                    return False
            else:
                log("Updating ML-Sharp repository...")
                subprocess.run(["git", "fetch", "origin"], cwd=str(sharp_path), timeout=60)
                result = subprocess.run(
                    ["git", "pull", "origin", "main"],
                    cwd=str(sharp_path),
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode != 0:
                    log(f"Pull failed: {result.stderr}")
                    # Continue anyway if pull fails but repo exists
            
            # 3. Create Virtual Environment
            venv_path = sharp_path / ".venv"
            if not venv_path.exists():
                log("Creating virtual environment...")
                py_args = python_exe.split()
                result = subprocess.run(
                    [*py_args, "-m", "venv", ".venv"],
                    cwd=str(sharp_path),
                    capture_output=True,
                    text=True,
                    timeout=120
                )
                if result.returncode != 0:
                    log(f"Venv creation failed: {result.stderr}")
                    return False
            
            # 4. Install Dependencies & Package
            log("Installing dependencies and the 'sharp' package...")
            # On Windows, PIP is in Scripts/pip.exe
            pip_exe = venv_path / "Scripts" / "pip.exe"
            
            # Ensure we're using the latest pip
            subprocess.run([str(pip_exe), "install", "--upgrade", "pip"], cwd=str(sharp_path), timeout=120)
            
            # Install CUDA-enabled torch and torchvision FIRST to prevent CPU versions from being pulled in
            log("Installing CUDA-enabled PyTorch (this involves a large download)...")
            subprocess.run(
                [str(pip_exe), "install", "torch", "torchvision", "--index-url", "https://download.pytorch.org/whl/cu124", "--force-reinstall"],
                cwd=str(sharp_path),
                timeout=600 # 10 minutes
            )

            # Install ml-sharp in editable mode - this is CRITICAL as it generates the 'sharp.exe' entry point
            log("Installing the 'sharp' package in editable mode...")
            result = subprocess.run(
                [str(pip_exe), "install", "-e", "."],
                cwd=str(sharp_path),
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                log(f"Package installation failed: {result.stderr}")
                return False
            
            # Verify the entry point was created
            sharp_exe = venv_path / "Scripts" / "sharp.exe"
            if not sharp_exe.exists():
                log("Error: 'sharp.exe' was not created. Attempting alternative installation...")
                subprocess.run([str(pip_exe), "install", "."], cwd=str(sharp_path), timeout=300)
                if not sharp_exe.exists():
                    log("Critical Error: Failed to generate 'sharp.exe' entry point.")
                    return False

            # Verify CUDA availability in the newly created venv
            log("Verifying CUDA availability...")
            verify_result = subprocess.run(
                [str(venv_path / "Scripts" / "python.exe"), "-c", "import torch; print(torch.cuda.is_available())"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if "True" not in verify_result.stdout:
                log("Warning: Torch was installed but CUDA is reported as not available.")
            else:
                log("CUDA availability verified!")

            # 5. Pre-download Model Checkpoint (Avoid SSL issues at runtime)
            checkpoint_dir = Path(os.path.expanduser("~")) / ".cache" / "torch" / "hub" / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / SHARP_CHECKPOINT_NAME
            
            if not checkpoint_path.exists():
                log(f"Downloading model checkpoint (~500MB)...")
                import ssl
                ssl_context = ssl._create_unverified_context()
                
                def progress(downloaded, total):
                    pct = int((downloaded / total) * 100)
                    if pct % 10 == 0:
                        log(f"Checkpoint download progress: {pct}%")
                
                try:
                    req = urllib.request.Request(SHARP_CHECKPOINT_URL)
                    req.add_header('User-Agent', 'AutomaticGaussianSplatter/1.0')
                    with urllib.request.urlopen(req, timeout=600, context=ssl_context) as response:
                        total_size = int(response.headers.get('Content-Length', 0))
                        downloaded = 0
                        with open(checkpoint_path, 'wb') as f:
                            while True:
                                chunk = response.read(16384)
                                if not chunk: break
                                f.write(chunk)
                                downloaded += len(chunk)
                                if total_size: progress(downloaded, total_size)
                    log("Checkpoint downloaded successfully!")
                except Exception as e:
                    log(f"Warning: Failed to pre-download checkpoint: {e}")
            
            # 6. Update version
            versions = self._load_versions()
            versions["sharp_commit"] = self._get_local_sharp_commit()
            self._save_versions(versions)
            
            log("ML-Sharp installed and configured successfully!")
            return True
            
        except Exception as e:
            log(f"ML-Sharp installation failed: {e}")
            logger.error(f"Sharp install error: {e}", exc_info=True)
            return False
    
    def _update_component(self, name: str, download_url: str, folder: str, 
                          version_key: str, latest_version: str, log) -> bool:
        """Update COLMAP or Brush component."""
        log(f"Starting {name} update to {latest_version}...")
        
        if not download_url:
            log(f"No download URL available for {name}")
            return False
        
        # Download to temp file
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            log(f"Downloading {name}...")
            
            def progress(downloaded, total):
                pct = int((downloaded / total) * 100)
                if pct % 10 == 0:
                    log(f"Download progress: {pct}%")
            
            if not self._download_file(download_url, tmp_path, progress):
                log(f"Failed to download {name}")
                return False
            
            log(f"Extracting and installing {name}...")
            if not self._extract_and_replace(tmp_path, folder):
                log(f"Failed to extract {name}")
                return False
            
            # Update version
            versions = self._load_versions()
            versions[version_key] = latest_version
            self._save_versions(versions)
            
            log(f"{name} updated successfully to {latest_version}!")
            return True
            
        finally:
            # Clean up temp file
            if tmp_path.exists():
                tmp_path.unlink()
    
    def _update_app(self, latest_commit: str, log) -> bool:
        """Update main app using git pull."""
        log("Updating Gaussian Splatter App via git pull...")
        
        try:
            # First fetch
            result = subprocess.run(
                ["git", "fetch", "origin"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                log(f"Git fetch failed: {result.stderr}")
                return False
            
            # Then pull
            result = subprocess.run(
                ["git", "pull", "origin", "main"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                # Try master branch if main doesn't exist
                result = subprocess.run(
                    ["git", "pull", "origin", "master"],
                    cwd=str(self.project_root),
                    capture_output=True,
                    text=True,
                    timeout=120
                )
            
            if result.returncode != 0:
                log(f"Git pull failed: {result.stderr}")
                return False
            
            # Update version
            new_commit = self._get_local_git_commit()
            versions = self._load_versions()
            versions["app_commit"] = new_commit
            self._save_versions(versions)
            
            log(f"App updated successfully to {new_commit[:8]}!")
            log("Note: You may need to restart the app for changes to take effect.")
            return True
            
        except subprocess.TimeoutExpired:
            log("Git operation timed out")
            return False
        except Exception as e:
            log(f"Git update failed: {e}")
            return False


# Singleton instance
_update_manager = None

def get_update_manager() -> UpdateManager:
    """Get or create the UpdateManager singleton."""
    global _update_manager
    if _update_manager is None:
        _update_manager = UpdateManager()
    return _update_manager
