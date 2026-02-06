#!/usr/bin/env python3
"""
Simple HTTP server to view the knowledge graph visualization.
Run this script and open the URL shown in your browser.
"""

import http.server
import socketserver
import webbrowser
import os
from pathlib import Path

# Configuration
PORT = 8000
GRAPH_FILE = "graph_viewer_simple.html"

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers to allow local file access
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        super().end_headers()

def find_available_port(start_port=8000, max_attempts=10):
    """Find an available port starting from start_port."""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return None

def main():
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Check if graph files exist
    if not Path('graph_data.json').exists():
        print("❌ Error: graph_data.json not found!")
        print("   Please run the graph creation script first.")
        return

    if not Path(GRAPH_FILE).exists():
        print(f"❌ Error: {GRAPH_FILE} not found!")
        return

    # Find available port
    port = find_available_port(PORT)
    if port is None:
        print(f"❌ Error: Could not find an available port between {PORT} and {PORT+10}")
        return

    if port != PORT:
        print(f"⚠ Port {PORT} was in use, using port {port} instead")

    # Start server
    with socketserver.TCPServer(("", port), MyHTTPRequestHandler) as httpd:
        url = f"http://localhost:{port}/{GRAPH_FILE}"
        print(f"\n{'='*60}")
        print(f"🚀 Knowledge Graph Viewer Server Started!")
        print(f"{'='*60}")
        print(f"\n📊 Open this URL in your browser:")
        print(f"   {url}")
        print(f"\n💡 Press Ctrl+C to stop the server")
        print(f"{'='*60}\n")

        # Try to open browser automatically
        try:
            webbrowser.open(url)
            print("✓ Browser opened automatically\n")
        except:
            print("⚠ Could not open browser automatically. Please open the URL manually.\n")

        # Serve forever
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Server stopped. Goodbye!")

if __name__ == "__main__":
    main()
