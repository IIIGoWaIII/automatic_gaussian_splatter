import asyncio
import os
import shutil

from pathlib import Path

async def run():
    repo_root = Path(__file__).parent.resolve()
    command = str(repo_root / "Helicon Focus 8" / "HeliconFocus.exe")
    output_image_path = str(repo_root / "test_debug_output.tif")
    stack_dir = str(repo_root / "test_debug_unpack" / "view_001")
    
    args = [
        "-silent",
        f"-save:\"{output_image_path}\"",
        f"\"{stack_dir}\""
    ]
    
    full_command = f'"{command}" ' + " ".join(args)
    print(f"Executing: {full_command}")
    
    process = await asyncio.create_subprocess_shell(
        full_command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    stdout, stderr = await process.communicate()
    
    print(f"Return Code: {process.returncode}")
    print(f"STDOUT: {stdout.decode(errors='replace')}")
    print(f"STDERR: {stderr.decode(errors='replace')}")

if __name__ == "__main__":
    asyncio.run(run())
