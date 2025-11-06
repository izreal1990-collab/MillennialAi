# VSCode Azure Debugging Setup for MillennialAi

This directory contains VSCode configuration files to help you debug and monitor the MillennialAi project on Azure.

## Files

- **tasks.json** - Defines tasks for monitoring Azure resources, running tests, and deploying
- **launch.json** - Defines debug configurations for Python applications
- **settings.json** - Python and Azure-specific settings for the workspace

## Azure Debugging Tasks

All tasks can be run from VSCode's Command Palette (`Ctrl+Shift+P` or `Cmd+Shift+P`) → "Tasks: Run Task"

### Monitoring Tasks

1. **Azure: Monitor All Resources**
   - Comprehensive monitoring of all Azure resources
   - Shows ML workspace, compute clusters, Container Apps, storage, and key vault
   - Checks for recent deployment errors
   - **Shortcut:** Most complete overview of your Azure deployment

2. **Azure: Monitor Continuous**
   - Continuously monitors Azure resources with periodic updates
   - Runs in background mode
   - Useful for watching resource state changes

3. **Azure: Check Container App Status**
   - Quick status check of Azure Container Apps
   - Shows deployment state and URLs

4. **Azure: View Container App Logs**
   - Streams live logs from the Azure Container App
   - Runs in background mode
   - Essential for debugging runtime issues

5. **Azure: Monitor ML Jobs**
   - Monitors Azure ML training jobs
   - Shows job status and progress

### Azure Resource Management Tasks

6. **Azure: Check ML Workspace**
   - Verifies Azure ML workspace status
   - Useful for validating workspace configuration

7. **Azure: List Compute Clusters**
   - Lists all compute clusters in the workspace
   - Shows cluster state, size, and configuration

8. **Azure: Check GPU Quota**
   - Checks GPU quota status
   - Important for planning GPU-based training

9. **Azure: Test Connectivity**
   - Tests Azure CLI authentication
   - Verifies subscription access

### Deployment Tasks

10. **Azure: Deploy to Container Apps**
    - Deploys the application to Azure Container Apps
    - Runs the deployment script

11. **Azure: View Recent Deployments**
    - Shows last 5 deployments and their status
    - Useful for tracking deployment history

12. **Azure: Check for Deployment Errors**
    - Analyzes failed deployments
    - Shows detailed error messages

13. **Azure: Open Portal Links**
    - Displays quick links to Azure Portal
    - Resource Group, Container Apps, and ML Studio URLs

### Local Debugging Tasks

14. **Debug: Run Python Tests**
    - Runs all unit tests with coverage
    - Default test task (`Ctrl+Shift+B` → Test)

15. **Debug: Run Quick Test**
    - Runs quick test to verify basic functionality
    - Faster than full test suite

16. **Debug: Test API Locally**
    - Starts FastAPI server locally
    - Runs on http://localhost:8000
    - Auto-reloads on code changes

17. **Debug: Lint Python Code**
    - Runs flake8 linter
    - Checks for syntax errors and code quality issues

18. **Debug: Install Dependencies**
    - Installs all required Python packages
    - Run this first if you're setting up the project

19. **Debug: Build Docker Image**
    - Builds Docker image locally for testing
    - Tagged as `millennialai:debug`

20. **Debug: Run Docker Container**
    - Runs the Docker container locally
    - Accessible on http://localhost:8000

## Debug Configurations

Available debug configurations (press `F5` or use Debug panel):

### Python Debugging

1. **Python: Current File**
   - Debug the currently open Python file
   - General-purpose debugging

2. **Python: FastAPI**
   - Debug the FastAPI application
   - Auto-reload enabled
   - Full debugging support with breakpoints

3. **Python: Quick Test**
   - Debug quick_test.py
   - Useful for testing core functionality

4. **Python: Pytest Current File**
   - Debug tests in the current file
   - Pytest-specific configuration

5. **Python: All Tests**
   - Debug all tests with coverage
   - Comprehensive testing

6. **Python: Azure ML Test**
   - Debug Azure ML integration
   - Requires Azure credentials in environment

7. **Python: Debug Injection Flow**
   - Debug the layer injection framework
   - Specialized for neural network debugging

8. **Docker: Attach to Container**
   - Attach debugger to running Docker container
   - Useful for debugging containerized applications

## Quick Start Guide

### First-Time Setup

1. Install dependencies:
   ```bash
   # Run task: "Debug: Install Dependencies"
   # Or manually:
   pip install -r requirements.txt
   ```

2. Test Azure connectivity:
   ```bash
   # Run task: "Azure: Test Connectivity"
   # Ensure you're logged in:
   az login
   ```

3. Verify basic functionality:
   ```bash
   # Run task: "Debug: Run Quick Test"
   ```

### Common Debugging Workflows

#### Debugging Local Development

1. Start the API locally:
   - Run task: "Debug: Test API Locally"
   - Or press `F5` → Select "Python: FastAPI"

2. Set breakpoints in your code

3. Test API endpoints using:
   - Browser: http://localhost:8000/docs
   - curl/Postman

#### Debugging Azure Deployment

1. Check current status:
   - Run task: "Azure: Monitor All Resources"

2. View live logs:
   - Run task: "Azure: View Container App Logs"

3. Check for errors:
   - Run task: "Azure: Check for Deployment Errors"

4. If deployment failed:
   - Fix the issue locally
   - Test with: "Debug: Build Docker Image" → "Debug: Run Docker Container"
   - Redeploy: "Azure: Deploy to Container Apps"

#### Debugging Azure ML Jobs

1. Monitor jobs:
   - Run task: "Azure: Monitor ML Jobs"

2. Check compute status:
   - Run task: "Azure: List Compute Clusters"

3. Check quota if GPU issues:
   - Run task: "Azure: Check GPU Quota"

### Environment Variables

For Azure ML debugging, set these environment variables:

```bash
export AZURE_SUBSCRIPTION_ID="your-subscription-id"
export AZURE_RESOURCE_GROUP="your-resource-group"
export AZURE_ML_WORKSPACE="your-workspace-name"
```

Or create a `.env` file in the workspace root (not committed to git).

## Troubleshooting

### Azure CLI Not Authenticated

```bash
az login
az account set --subscription <your-subscription-id>
```

### Python Dependencies Missing

```bash
# Run task: "Debug: Install Dependencies"
```

### Docker Build Fails

1. Check Dockerfile syntax
2. Ensure all dependencies are in requirements.txt
3. Run: "Debug: Lint Python Code" to check for errors

### Container App Not Responding

1. Run: "Azure: View Container App Logs"
2. Check for startup errors
3. Verify environment variables in Azure Portal

## Tips

- **Use keyboard shortcuts**: Assign keyboard shortcuts to frequently used tasks
- **Terminal integration**: Tasks run in integrated terminal for easy access
- **Background tasks**: Some tasks (like log streaming) run in background
- **Problem matcher**: Python tasks have problem matchers for error highlighting

## Additional Resources

- [Azure Portal](https://portal.azure.com/)
- [Azure ML Studio](https://ml.azure.com/)
- [VSCode Python Debugging](https://code.visualstudio.com/docs/python/debugging)
- [Azure Container Apps Docs](https://learn.microsoft.com/en-us/azure/container-apps/)
