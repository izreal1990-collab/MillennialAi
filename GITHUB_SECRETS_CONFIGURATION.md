# GitHub Secrets Configuration

## Azure Service Principal Details
- **Service Principal Name**: GitHub-MillennialAi-Actions-1762114115
- **Client ID**: d1f5a611-eee3-485d-bcb2-7efb502260b6
- **Status**: Created and tested successfully
- **Permissions**: Contributor and AzureML Data Scientist roles assigned

## Required GitHub Secrets

Navigate to: https://github.com/izreal1990-collab/MillennialAi/settings/secrets/actions

Configure these 5 secrets:

### 1. AZURE_CREDENTIALS
```json
{
  "clientId": "d1f5a611-eee3-485d-bcb2-7efb502260b6",
  "clientSecret": "[Generated during service principal creation]",
  "subscriptionId": "2f04fc1c-4e39-48a7-b9b3-43ea2b6a593e",
  "tenantId": "e1c4c7a3-6d7a-4c7a-a4d7-8f9e3c5d7a6b"
}
```

### 2. AZURE_SUBSCRIPTION_ID
```
2f04fc1c-4e39-48a7-b9b3-43ea2b6a593e
```

### 3. AZURE_RESOURCE_GROUP
```
MillennialAi
```

### 4. AZURE_ML_WORKSPACE
```
millennialai-workspace
```

### 5. AZURE_FOUNDRY_HUB
```
millennialai-foundry-hub
```

## Next Steps
1. Configure all 5 secrets in GitHub
2. Re-enable Azure validation job in workflow
3. Test complete CI/CD pipeline

## Validation Commands
After configuration, the workflow will:
- Authenticate with Azure using service principal
- Validate ML workspace access
- Check AI Foundry hub status
- Monitor GPU quota availability