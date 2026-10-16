#!/usr/bin/env python
"""Run the Autonomous AI Trading Platform Web Dashboard & Background Scheduler."""

import argparse
import os
import sys

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from analyst.web_app import run_web_server


def main():
    parser = argparse.ArgumentParser(description="Start the AlphaNexus AI Trading Platform Web App")
    parser.add_argument("--port", type=int, default=None, help="Port to listen on (default: $PORT or 8050)")
    parser.add_argument("--no-scheduler", action="store_true", help="Disable the background scheduler daemon")
    args = parser.parse_args()

    port = args.port or int(os.environ.get("PORT", 8050))
    server = run_web_server(port=port, enable_scheduler=not args.no_scheduler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping web server...")
        server.shutdown()
        sys.exit(0)


if __name__ == "__main__":
    main()
