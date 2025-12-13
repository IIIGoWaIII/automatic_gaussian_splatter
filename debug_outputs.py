import sys
import os
from pathlib import Path

# Add python_webui to path
sys.path.append(os.path.join(os.getcwd(), 'python_webui'))

from pipeline_manager import PipelineManager

def debug_outputs():
    base_dir = Path(r"d:\software\gaussian_splatter\python_webui\processing_output")
    print(f"Checking base dir: {base_dir}")
    print(f"Exists: {base_dir.exists()}")
    
    manager = PipelineManager(str(base_dir))
    outputs = manager.get_available_outputs()
    
    print(f"Found {len(outputs)} outputs:")
    for out in outputs:
        print(f" - {out['folder']} (Checkpoints: {len(out['ply_checkpoints'])})")

if __name__ == "__main__":
    debug_outputs()
