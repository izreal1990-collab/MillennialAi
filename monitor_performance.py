#!/usr/bin/env python3
"""
MillennialAi Performance Monitoring Script
Generates constantly updating performance report and commits to git
"""

import os
import json
import time
import requests
import subprocess
from datetime import datetime
from pathlib import Path
from continuous_learning import continuous_learning

class PerformanceMonitor:
    """Monitors MillennialAi system performance"""

    def __init__(self, api_url="http://localhost:8001", report_file="PERFORMANCE_REPORT.md"):
        self.api_url = api_url
        self.report_file = Path(report_file)
        self.last_commit_time = None

    def get_api_stats(self):
        """Get API performance stats"""
        try:
            # Health check
            health_response = requests.get(f"{self.api_url}/health", timeout=5)
            health_data = health_response.json() if health_response.status_code == 200 else {}

            # Get learning stats for conversation count
            try:
                learning_response = requests.get(f"{self.api_url}/learning/stats", timeout=5)
                learning_data = learning_response.json() if learning_response.status_code == 200 else {}
                conv_count = learning_data.get('total_conversations', 0)
            except Exception:
                conv_count = 0

            return {
                'status': 'running' if health_response.status_code == 200 else 'down',
                'response_time': health_response.elapsed.total_seconds() if hasattr(health_response, 'elapsed') else 0,
                'conversations': conv_count,
                'uptime': health_data.get('uptime', 'unknown'),
                'version': health_data.get('version', 'unknown')
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'response_time': 0,
                'conversations': 0,
                'uptime': 'unknown',
                'version': 'unknown'
            }

    def get_system_stats(self):
        """Get basic system stats"""
        PSUTIL_NOT_INSTALLED = 'psutil not installed'
        try:
            # CPU usage (simple)
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_total_gb': memory.total / (1024**3)
            }
        except ImportError:
            return {
                'cpu_percent': PSUTIL_NOT_INSTALLED,
                'memory_percent': PSUTIL_NOT_INSTALLED,
                'memory_used_gb': PSUTIL_NOT_INSTALLED,
                'memory_total_gb': PSUTIL_NOT_INSTALLED
            }

    def generate_report(self):
        """Generate performance report"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

        # Get all stats
        api_stats = self.get_api_stats()
        learning_stats = continuous_learning.get_learning_stats()
        system_stats = self.get_system_stats()

        # Create report
        report = f"""# MillennialAi Performance Report
*Generated: {timestamp}*

## System Status
- **API Status**: {api_stats['status']}
- **Response Time**: {api_stats['response_time']:.2f}s
- **Uptime**: {api_stats['uptime']}
- **Version**: {api_stats['version']}

## Continuous Learning Stats
- **Total Samples Collected**: {learning_stats['total_samples_collected']}
- **Average Sample Quality**: {learning_stats['avg_sample_quality']:.3f}
- **Retraining Jobs Submitted**: {learning_stats['retraining_jobs_submitted']}
- **Last Retraining Samples**: {learning_stats['last_retraining_samples']}
- **Next Retraining Eligible**: {learning_stats['next_retraining_eligible']}

## Conversations
- **Total Conversations**: {api_stats['conversations']}

## System Resources
- **CPU Usage**: {system_stats['cpu_percent']}%
- **Memory Usage**: {system_stats['memory_percent']}%
- **Memory Used**: {system_stats['memory_used_gb']:.2f} GB
- **Memory Total**: {system_stats['memory_total_gb']:.2f} GB

## Recent Activity
- **Last Update**: {timestamp}

---
*This report updates every 5 minutes. Auto-committed to git.*
"""

        return report

    def update_report(self):
        """Update the report file"""
        report_content = self.generate_report()

        # Write to file
        with open(self.report_file, 'w') as f:
            f.write(report_content)

        print(f"✅ Report updated at {datetime.now().strftime('%H:%M:%S')}")

    def commit_to_git(self):
        """Commit report to git if changed"""
        try:
            # Check if file changed
            result = subprocess.run(['git', 'status', '--porcelain', str(self.report_file)],
                                  capture_output=True, text=True, cwd=Path.cwd())

            if result.stdout.strip():  # File changed
                # Add file
                subprocess.run(['git', 'add', str(self.report_file)], check=True, cwd=Path.cwd())

                # Commit
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                commit_msg = f"Auto-update performance report - {timestamp}"
                subprocess.run(['git', 'commit', '-m', commit_msg], check=True, cwd=Path.cwd())

                # Push (optional, comment out if not wanted)
                # subprocess.run(['git', 'push'], check=True, cwd=Path.cwd())

                print(f"✅ Report committed to git at {datetime.now().strftime('%H:%M:%S')}")
                self.last_commit_time = datetime.now()

        except subprocess.CalledProcessError as e:
            print(f"⚠️ Git commit failed: {e}")
        except Exception as e:
            print(f"⚠️ Git operation error: {e}")

def main():
    """Main monitoring loop"""
    monitor = PerformanceMonitor()

    print("🚀 Starting MillennialAi Performance Monitor")
    print("📊 Report will update every 5 minutes")
    print("💾 Auto-committing to git when changed")

    while True:
        try:
            # Update report
            monitor.update_report()

            # Commit to git
            monitor.commit_to_git()

            # Wait 5 minutes
            time.sleep(300)

        except KeyboardInterrupt:
            print("\n🛑 Monitor stopped by user")
            break
        except Exception as e:
            print(f"⚠️ Monitor error: {e}")
            time.sleep(60)  # Wait a minute before retrying

if __name__ == "__main__":
    main()