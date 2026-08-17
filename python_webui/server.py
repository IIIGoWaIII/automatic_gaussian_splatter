import sys
import asyncio
import uvicorn
import webbrowser
import threading
import time
import subprocess
import re
from main import app

def open_browser():
    """Wait for the server to start, then open the browser."""
    # Give the server a moment to start
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:8000")

def get_network_ips():
    """Return IPv4 addresses of real and virtual (e.g. Tailscale) adapters,
    excluding loopback and common virtual/private networks."""
    try:
        ps = (
            "(Get-NetIPAddress -AddressFamily IPv4 | "
            "Where-Object { $_.IPAddress -notlike '169.254.*' -and "
            "$_.InterfaceAlias -notmatch 'Loopback|Hyper-V|vEthernet|WSL|Docker|Bluetooth|VirtualBox|VMware|Hamachi' } | "
            "Select-Object -ExpandProperty IPAddress)"
        )
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=15
        )
        return [ip.strip() for ip in out.stdout.splitlines()
                if re.match(r'^\d+\.\d+\.\d+\.\d+$', ip.strip())]
    except Exception:
        return []

if __name__ == "__main__":
    # Force ProactorEventLoop on Windows for subprocess support
    # This addresses the NotImplementedError in asyncio.create_subprocess_shell
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Start browser in a separate thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run uvicorn without reload to ensure the loop policy is respected and stable
    print("Starting server with WindowsProactorEventLoopPolicy...")
    print("Local:   http://127.0.0.1:8000")
    for ip in get_network_ips():
        print(f"Network: http://{ip}:8000  (LAN / Tailnet)")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")