import sys
import asyncio
import uvicorn
from main import app

if __name__ == "__main__":
    # Force ProactorEventLoop on Windows for subprocess support
    # This addresses the NotImplementedError in asyncio.create_subprocess_shell
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Run uvicorn without reload to ensure the loop policy is respected and stable
    print("Starting server with WindowsProactorEventLoopPolicy...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
