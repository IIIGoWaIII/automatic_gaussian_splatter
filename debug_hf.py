import asyncio
import os
import shutil

async def run():
    command = r"d:\software\gaussian_splatter\Helicon Focus 8\HeliconFocus.exe"
    output_image_path = r"d:\software\gaussian_splatter\test_debug_output.tif"
    stack_dir = r"d:\software\gaussian_splatter\test_debug_unpack\view_001"
    
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
