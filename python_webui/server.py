import sys
import asyncio
import uvicorn
import webbrowser
import threading
import time
from main import app

def open_browser():
    """Wait for the server to start, then open the browser."""
    # Give the server a moment to start
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    # Force ProactorEventLoop on Windows for subprocess support
    # This addresses the NotImplementedError in asyncio.create_subprocess_shell
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Start browser in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run uvicorn without reload to ensure the loop policy is respected and stable
    print("Starting server with WindowsProactorEventLoopPolicy...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
