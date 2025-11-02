# GitHub Integration Setup for MillennialAi

This guide helps you configure GitHub Actions for seamless Azure integration with your MillennialAi project.

## 🔐 Required GitHub Secrets

You need to set up the following secrets in your GitHub repository:

### Navigate to GitHub Secrets
1. Go to your repository: https://github.com/izreal1990-collab/MillennialAi
2. Settings → Secrets and variables → Actions
3. Click "New repository secret" for each of the following:

### Azure Authentication
```
Name: AZURE_CREDENTIALS
Value: {
  "clientId": "YOUR_SERVICE_PRINCIPAL_CLIENT_ID",
  "clientSecret": "YOUR_SERVICE_PRINCIPAL_SECRET",
  "subscriptionId": "639e13e9-b4be-4ba6-8e9e-f14db5b3a65c",
  "tenantId": "YOUR_TENANT_ID"
}
```

### Azure Resource Configuration
```
Name: AZURE_SUBSCRIPTION_ID
Value: 639e13e9-b4be-4ba6-8e9e-f14db5b3a65c

Name: AZURE_RESOURCE_GROUP  
Value: rg-jblango-1749

Name: AZURE_ML_WORKSPACE
Value: azml7c462q3plqeyo

Name: AZURE_FOUNDRY_HUB
Value: millennialai-foundry-hub
```

## 🛠️ Create Azure Service Principal

Run these commands to create the service principal for GitHub Actions:

```bash
# Create service principal
az ad sp create-for-rbac \
  --name "GitHub-MillennialAi-Actions" \
  --role "Contributor" \
  --scopes "/subscriptions/639e13e9-b4be-4ba6-8e9e-f14db5b3a65c/resourceGroups/rg-jblango-1749" \
  --sdk-auth

# Add additional roles for AI services
az role assignment create \
  --assignee YOUR_SERVICE_PRINCIPAL_CLIENT_ID \
  --role "Cognitive Services Contributor" \
  --scope "/subscriptions/639e13e9-b4be-4ba6-8e9e-f14db5b3a65c/resourceGroups/rg-jblango-1749"

az role assignment create \
  --assignee YOUR_SERVICE_PRINCIPAL_CLIENT_ID \
  --role "Machine Learning Workspace Contributor" \
  --scope "/subscriptions/639e13e9-b4be-4ba6-8e9e-f14db5b3a65c/resourceGroups/rg-jblango-1749"
```

## 🚀 GitHub Actions Features

### Automatic Testing
- **Python compatibility**: Tests on Python 3.8, 3.9, 3.10
- **MillennialAi validation**: Imports and basic functionality
- **Code quality**: Linting with flake8
- **Coverage**: Test coverage reports

### Azure Integration
- **Resource validation**: Checks Azure ML workspace and compute clusters
- **GPU quota monitoring**: Automatic quota status checks
- **Deployment**: Optional training job submission
- **AI Foundry**: Validates hub and project connectivity

### Workflow Triggers
- **Push to main/develop**: Full test and validation
- **Pull requests**: Testing and validation
- **Manual dispatch**: Optional Azure deployment
- **GPU quota checks**: Automated monitoring

## 📊 Workflow Status

The workflow provides:
- ✅ Code quality validation
- ✅ Azure resource health checks  
- ✅ GPU quota status monitoring
- ✅ Deployment readiness validation
- ✅ AI Foundry integration testing

## 🔧 Quick Setup Commands

Once you have the service principal output, run this to set up secrets quickly:

```bash
# Save the service principal output to a file
# Then copy each value to GitHub secrets as shown above

# Test the setup locally first:
az login --service-principal \
  --username YOUR_CLIENT_ID \
  --password YOUR_CLIENT_SECRET \
  --tenant YOUR_TENANT_ID

# Verify access
az ml workspace show \
  --name azml7c462q3plqeyo \
  --resource-group rg-jblango-1749
```

## 🎯 Benefits

With GitHub Actions setup, you get:

1. **Continuous Integration**: Every commit tested
2. **Azure Validation**: Resources monitored automatically  
3. **GPU Monitoring**: Know when quota is approved
4. **Deployment Automation**: One-click training job submission
5. **Quality Assurance**: Code quality and testing enforcement
6. **Documentation**: Automatic workflow status updates

## 📞 Troubleshooting

### Common Issues:
1. **Service Principal Permissions**: Ensure all required roles assigned
2. **Secret Format**: AZURE_CREDENTIALS must be valid JSON
3. **Resource Names**: Double-check workspace and resource group names
4. **Quota Status**: GPU workflow helps monitor quota approval

### Validation Commands:
```bash
# Test service principal locally
az account show

# Validate ML workspace access  
az ml workspace list --resource-group rg-jblango-1749

# Check compute clusters
az ml compute list --workspace-name azml7c462q3plqeyo --resource-group rg-jblango-1749
```

---

**Next Steps**: After setting up secrets, push any commit to main branch to trigger the workflow and validate the integration!