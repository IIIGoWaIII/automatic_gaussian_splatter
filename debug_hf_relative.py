import asyncio
import os
import shutil

async def run():
    command = r"d:\software\gaussian_splatter\Helicon Focus 8\HeliconFocus.exe"
    # Use relative path by changing cwd
    base_dir = r"D:\software\gaussian_splatter\python_webui\processing_output\aef8a3c3-88fe-41df-9be7-ecf834e8e08c"
    os.chdir(base_dir)
    
    output_image_rel = r"images\view_001.tif"
    stack_dir_rel = r"input_source\view_001"
    
    # Try without quotes since no spaces in relative path
    args = [
        "-silent",
        f"-save:{output_image_rel}",
        f"{stack_dir_rel}"
    ]
    
    full_command = f'"{command}" ' + " ".join(args)
    print(f"CWD: {os.getcwd()}")
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
    
    if os.path.exists(output_image_rel):
        print("SUCCESS: Output file created.")
    else:
        print("FAILURE: Output file NOT created.")

if __name__ == "__main__":
    asyncio.run(run())
