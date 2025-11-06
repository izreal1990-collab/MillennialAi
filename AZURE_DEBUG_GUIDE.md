# Azure Debugging Guide for MillennialAi

This guide explains how to debug the MillennialAi project on Azure using the VSCode tasks and debug configurations.

## Quick Start

### Prerequisites

1. **VSCode** installed with Python extension
2. **Azure CLI** installed and authenticated (`az login`)
3. **Python 3.8+** installed
4. **Docker** installed (optional, for container debugging)

### First-Time Setup

1. Open the project in VSCode:
   ```bash
   code /path/to/MillennialAi
   ```

2. Install Python dependencies:
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Tasks: Run Task"
   - Select "Debug: Install Dependencies"

3. Verify Azure authentication:
   - Run task: "Azure: Test Connectivity"

## Using the Debugging Tasks

### Viewing Available Tasks

Press `Ctrl+Shift+P` → "Tasks: Run Task" to see all 21 available tasks.

### Most Useful Tasks

#### For Azure Monitoring
- **Azure: Monitor All Resources** - Get complete status of all Azure resources
- **Azure: View Container App Logs** - Stream live logs from your deployed app
- **Azure: Check for Deployment Errors** - Diagnose failed deployments

#### For Local Development
- **Debug: Test API Locally** - Start FastAPI server with auto-reload
- **Debug: Run Python Tests** - Run all tests with coverage
- **Debug: Build Docker Image** - Build and test Docker container locally

#### For Deployment
- **Azure: Deploy to Container Apps** - Deploy your app to Azure

### Using Debug Configurations

Press `F5` to start debugging with these configurations:

1. **Python: FastAPI** - Debug the web API with breakpoints
2. **Python: Current File** - Debug any Python file
3. **Python: All Tests** - Debug tests with coverage

## Common Workflows

### 1. Debugging a Failed Azure Deployment

```
Step 1: Check deployment status
→ Run task: "Azure: Monitor All Resources"

Step 2: View error details
→ Run task: "Azure: Check for Deployment Errors"

Step 3: Check live logs
→ Run task: "Azure: View Container App Logs"

Step 4: Fix locally and test
→ Run task: "Debug: Build Docker Image"
→ Run task: "Debug: Run Docker Container"

Step 5: Redeploy
→ Run task: "Azure: Deploy to Container Apps"
```

### 2. Debugging API Issues Locally

```
Step 1: Start API in debug mode
→ Press F5 → Select "Python: FastAPI"

Step 2: Set breakpoints in your code

Step 3: Test API endpoints
→ Open http://localhost:8000/docs

Step 4: Inspect variables when breakpoint hits
```

### 3. Monitoring Azure Resources

```
Continuous monitoring:
→ Run task: "Azure: Monitor Continuous"

Check specific resources:
→ "Azure: Check Container App Status"
→ "Azure: List Compute Clusters"
→ "Azure: Check GPU Quota"
```

## Task Reference

### Azure Monitoring Tasks (9)
1. Azure: Monitor All Resources
2. Azure: Monitor Continuous
3. Azure: Check Container App Status
4. Azure: View Container App Logs
5. Azure: Check ML Workspace
6. Azure: List Compute Clusters
7. Azure: Check GPU Quota
8. Azure: Test Connectivity
9. Azure: Monitor ML Jobs

### Azure Deployment Tasks (4)
10. Azure: Deploy to Container Apps
11. Azure: View Recent Deployments
12. Azure: Check for Deployment Errors
13. Azure: Open Portal Links

### Local Development Tasks (8)
14. Debug: Run Python Tests
15. Debug: Run Quick Test
16. Debug: Test API Locally
17. Debug: Lint Python Code
18. Debug: Install Dependencies
19. Debug: Build Docker Image
20. Debug: Run Docker Container

## Environment Variables

For Azure ML operations, set these environment variables:

```bash
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="your-resource-group"
export AZURE_ML_WORKSPACE="your-workspace-name"
```

## Keyboard Shortcuts

Assign custom keyboard shortcuts to frequently used tasks:

1. `File` → `Preferences` → `Keyboard Shortcuts`
2. Search for "Tasks: Run Task"
3. Add keybinding (e.g., `Ctrl+Alt+M` for monitoring)

## Troubleshooting

### Azure CLI Authentication Error
```bash
az login
az account set --subscription <your-subscription-id>
```

### Task Not Found Error
Ensure you're in the workspace root directory where `.vscode/tasks.json` exists.

### Python Module Not Found
Run task: "Debug: Install Dependencies"

### Docker Permission Denied
```bash
sudo usermod -aG docker $USER
# Logout and login again
```

## Additional Resources

- **Detailed Documentation**: See `.vscode/README.md`
- **Azure Portal**: https://portal.azure.com/
- **Azure ML Studio**: https://ml.azure.com/
- **VSCode Debugging Guide**: https://code.visualstudio.com/docs/editor/debugging

## Getting Help

If you encounter issues:

1. Check the `.vscode/README.md` for detailed documentation
2. Run diagnostic tasks to identify the problem
3. Check Azure Portal for service-specific issues
4. Review logs using the log viewing tasks

## Tips

- Use `Ctrl+Shift+B` to quickly access build/test tasks
- Background tasks (log streaming) can run while you work
- Set breakpoints before starting debug configurations
- Use task output terminal for detailed error messages
- Chain tasks together using shell scripts if needed

---

**Note**: This debugging setup is specifically designed for the MillennialAi project's Azure deployment. Tasks reference specific Azure resource names and configurations from the project.
