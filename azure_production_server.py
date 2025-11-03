#!/usr/bin/env python3
"""
MillennialAi Azure Production Server
Runs both the live chat API and continuous learning system for continuous operation
"""

import os
import sys
import time
import signal
import threading
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AzureProductionServer:
    """Manages the production server with both API and continuous learning"""

    def __init__(self):
        self.api_process = None
        self.learning_process = None
        self.running = True

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.shutdown)
        signal.signal(signal.SIGINT, self.shutdown)

    def start_api_server(self):
        """Start the live chat API server"""
        try:
            logger.info("🚀 Starting MillennialAi Live Chat API...")

            # Set environment variables for API
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{env.get('PYTHONPATH', '')}:{os.getcwd()}"
            env['HOST'] = '0.0.0.0'
            env['PORT'] = '8000'

            self.api_process = subprocess.Popen([
                sys.executable, 'millennial_ai_live_chat.py'
            ], env=env, cwd=os.getcwd())

            logger.info("✅ Live Chat API started successfully")

        except Exception as e:
            logger.error(f"❌ Failed to start API server: {e}")
            raise

    def start_continuous_learning(self):
        """Start the continuous learning system"""
        try:
            logger.info("🧠 Starting Continuous Learning System (90% Capacity)...")

            # Set environment variables for continuous learning
            env = os.environ.copy()
            env['PYTHONPATH'] = f"{env.get('PYTHONPATH', '')}:{os.getcwd()}"

            self.learning_process = subprocess.Popen([
                sys.executable, 'continuous_learning.py'
            ], env=env, cwd=os.getcwd())

            logger.info("✅ Continuous Learning System started successfully")

        except Exception as e:
            logger.error(f"❌ Failed to start continuous learning: {e}")
            raise

    def monitor_processes(self):
        """Monitor both processes and restart if they fail"""
        while self.running:
            try:
                # Check API process
                if self.api_process and self.api_process.poll() is not None:
                    logger.warning("⚠️ API process died, restarting...")
                    self.start_api_server()

                # Check learning process
                if self.learning_process and self.learning_process.poll() is not None:
                    logger.warning("⚠️ Continuous learning process died, restarting...")
                    self.start_continuous_learning()

                time.sleep(30)  # Check every 30 seconds

            except Exception as e:
                logger.error(f"❌ Error in process monitoring: {e}")
                time.sleep(10)

    def shutdown(self, signum=None, frame=None):
        """Shutdown all processes gracefully"""
        logger.info("🛑 Shutting down Azure Production Server...")

        self.running = False

        # Terminate processes
        for process_name, process in [("API", self.api_process), ("Learning", self.learning_process)]:
            if process and process.poll() is None:
                logger.info(f"Stopping {process_name} process...")
                try:
                    process.terminate()
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning(f"Force killing {process_name} process...")
                    process.kill()

        logger.info("👋 Azure Production Server shutdown complete")
        sys.exit(0)

    def run(self):
        """Main run loop"""
        logger.info("🌐 Starting MillennialAi Azure Production Server")
        logger.info("📊 System configured for 90% resource utilization")
        logger.info("=" * 60)

        try:
            # Start both services
            self.start_api_server()
            time.sleep(2)  # Brief delay between startups
            self.start_continuous_learning()

            logger.info("✅ All services started successfully")
            logger.info("🌐 API available at: http://0.0.0.0:8000")
            logger.info("🧠 Continuous learning running at 90% capacity")
            logger.info("⏱️ Monitoring processes every 30 seconds")
            logger.info("=" * 60)

            # Start monitoring
            self.monitor_processes()

        except Exception as e:
            logger.error(f"❌ Failed to start production server: {e}")
            self.shutdown()

if __name__ == "__main__":
    server = AzureProductionServer()
    server.run()