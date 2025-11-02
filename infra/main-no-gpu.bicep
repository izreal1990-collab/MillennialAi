// Target scope
targetScope = 'resourceGroup'

// Parameters
@minLength(1)
@maxLength(64)
@description('Name of the environment which is used to generate a short unique hash for resources.')
param environmentName string

@minLength(1)
@description('Primary location for all resources')
param location string = resourceGroup().location

// Variables
var resourceToken = uniqueString(subscription().id, resourceGroup().id, location, environmentName)

// User-Assigned Managed Identity (UAMI) - Required by azd rules
resource userAssignedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'azuai${resourceToken}'
  location: location
}

// Log Analytics Workspace
resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: 'azlaw${resourceToken}'
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

// Container Registry
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: 'azacr${resourceToken}'
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
  }
}

// Role Assignment for ACR Pull
resource acrPullRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(containerRegistry.id, userAssignedIdentity.id, '7f951dda-4ed3-4680-a7ca-43fe172d538d')
  scope: containerRegistry
  properties: {
    principalId: userAssignedIdentity.properties.principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
    principalType: 'ServicePrincipal'
  }
}

// Container Apps Environment
resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: 'azcae${resourceToken}'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
  }
}

// Storage Account for training data and models
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: 'azsa${resourceToken}'
  location: location
  kind: 'StorageV2'
  sku: {
    name: 'Standard_LRS'
  }
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
  }
}

// Blob containers for training artifacts
resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  name: 'default'
  parent: storageAccount
}

resource trainingDataContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: 'training-data'
  parent: blobService
  properties: {
    publicAccess: 'None'
  }
}

resource modelsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: 'models'
  parent: blobService
  properties: {
    publicAccess: 'None'
  }
}

resource checkpointsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: 'checkpoints'
  parent: blobService
  properties: {
    publicAccess: 'None'
  }
}

// Key Vault for secrets and credentials
resource keyVault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'azkv${resourceToken}'
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
  }
}

// Application Insights for monitoring training
resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'azai${resourceToken}'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logAnalytics.id
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

// Azure Machine Learning Workspace
resource mlWorkspace 'Microsoft.MachineLearningServices/workspaces@2024-04-01' = {
  name: 'azml${resourceToken}'
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    friendlyName: 'MillennialAi ML Workspace'
    description: 'Azure ML workspace for MillennialAi training and experimentation'
    storageAccount: storageAccount.id
    keyVault: keyVault.id
    applicationInsights: applicationInsights.id
    containerRegistry: containerRegistry.id
    publicNetworkAccess: 'Enabled'
  }
}

// CPU Compute Cluster for data preprocessing (no GPU quota needed)
resource cpuComputeCluster 'Microsoft.MachineLearningServices/workspaces/computes@2024-04-01' = {
  name: 'cpu-cluster'
  parent: mlWorkspace
  location: location
  properties: {
    computeType: 'AmlCompute'
    properties: {
      vmSize: 'Standard_DS3_v2'
      vmPriority: 'Dedicated'
      scaleSettings: {
        minNodeCount: 0
        maxNodeCount: 2
        nodeIdleTimeBeforeScaleDown: 'PT120S'
      }
      remoteLoginPortPublicAccess: 'Disabled'
    }
  }
}

// Role assignment for ML workspace to access storage
resource mlStorageRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(storageAccount.id, mlWorkspace.id, 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
  scope: storageAccount
  properties: {
    principalId: mlWorkspace.identity.principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
    principalType: 'ServicePrincipal'
  }
}

// Role assignment for user identity to access Key Vault (Key Vault Secrets User role)
resource kvRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(keyVault.id, userAssignedIdentity.id, '4633458b-17de-408a-b874-0445c86b69e6')
  scope: keyVault
  properties: {
    principalId: userAssignedIdentity.properties.principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')
    principalType: 'ServicePrincipal'
  }
}

// Container App for MillennialAi (using basic config, no Docker build)
resource millennialaiApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'azca${resourceToken}'
  location: location
  tags: {
    'azd-service-name': 'millennialai'
  }
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${userAssignedIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: containerAppsEnvironment.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        corsPolicy: {
          allowedOrigins: ['*']
          allowedMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
          allowedHeaders: ['*']
        }
      }
      registries: [
        {
          server: containerRegistry.properties.loginServer
          identity: userAssignedIdentity.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'millennialai'
          image: 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            {
              name: 'PYTHONPATH'
              value: '/app'
            }
            {
              name: 'PORT'
              value: '8000'
            }
            {
              name: 'AZURE_ENV_NAME'
              value: environmentName
            }
            {
              name: 'AZURE_ML_WORKSPACE'
              value: mlWorkspace.name
            }
            {
              name: 'AZURE_STORAGE_ACCOUNT'
              value: storageAccount.name
            }
            {
              name: 'APPLICATION_INSIGHTS_CONNECTION_STRING'
              value: applicationInsights.properties.ConnectionString
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 3
      }
    }
  }
  dependsOn: [
    acrPullRoleAssignment
  ]
}

// Outputs (Required by azd rules)
output RESOURCE_GROUP_ID string = resourceGroup().id
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = containerRegistry.properties.loginServer
output AZURE_ML_WORKSPACE_NAME string = mlWorkspace.name
output AZURE_STORAGE_ACCOUNT_NAME string = storageAccount.name
output AZURE_KEY_VAULT_NAME string = keyVault.name
output APPLICATION_INSIGHTS_CONNECTION_STRING string = applicationInsights.properties.ConnectionString
output AZURE_ML_WORKSPACE_ID string = mlWorkspace.id
