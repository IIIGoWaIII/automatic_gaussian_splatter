import asyncio
import os
import shutil

async def run():
    command = r"d:\software\gaussian_splatter\Helicon Focus 8\HeliconFocus.exe"
    # Exact path from log
    output_image_path = r"D:\software\gaussian_splatter\python_webui\processing_output\aef8a3c3-88fe-41df-9be7-ecf834e8e08c\images\view_001.tif"
    stack_dir = r"D:\software\gaussian_splatter\python_webui\processing_output\aef8a3c3-88fe-41df-9be7-ecf834e8e08c\input_source\view_001"
    
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
    
    # Check if file exists
    if os.path.exists(output_image_path):
        print("SUCCESS: Output file created.")
    else:
        print("FAILURE: Output file NOT created.")

if __name__ == "__main__":
    asyncio.run(run())
