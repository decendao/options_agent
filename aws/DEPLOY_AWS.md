# AWS Deployment Guide

Deploy Options Agent to AWS using ECS Fargate (recommended) or EC2.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    ECS Fargate                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Options Agent Container (0.5 vCPU / 1GB)      │    │
│  │  • Polls Alpaca Market Data every 30s           │    │
│  │  • Computes Greeks / IV Rank / Gamma Profile    │    │
│  │  • Dispatches alerts via webhook / Slack        │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
          │                              │
          ▼                              ▼
┌─────────────────────┐    ┌─────────────────────────────────┐
│  Secrets Manager    │    │  CloudWatch Logs                │
│  • ALPACA_API_KEY  │    │  • Structured JSON logs         │
│  • ALPACA_SECRET   │    │  • 14-day retention             │
│  • POLYGON_API_KEY │    └─────────────────────────────────┘
└─────────────────────┘

Estimated monthly cost: ~$17-20 (Fargate) or ~$10-13 (EC2 t3.micro)
Data sources: FREE (Alpaca Free Tier + Polygon Starter + Polymarket/Kalshi public APIs)
```

---

## Prerequisites

- AWS CLI configured (`aws configure`)
- Docker installed locally
- An AWS account with appropriate IAM permissions
- Alpaca API keys: https://app.alpaca.markets
- Polygon.io API key: https://polygon.io

---

## Step 1 — Set Up AWS Resources

### 1.1 Create ECR Repository

```bash
aws ecr create-repository \
  --repository-name options-agent \
  --region us-east-1

# Note the repository URI, e.g.:
# YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/options-agent
```

### 1.2 Create Secrets in Secrets Manager

```bash
# Store Alpaca API credentials
aws secretsmanager create-secret \
  --name options-agent/alpaca_key \
  --secret-string '{"ALPACA_API_KEY":"your_key_id"}' \
  --region us-east-1

aws secretsmanager create-secret \
  --name options-agent/alpaca_secret \
  --secret-string '{"ALPACA_API_SECRET":"your_secret"}' \
  --region us-east-1

# Store Polygon API key
aws secretsmanager create-secret \
  --name options-agent/polygon_key \
  --secret-string '{"POLYGON_API_KEY":"your_polygon_key"}' \
  --region us-east-1
```

### 1.3 Create CloudWatch Log Group

```bash
aws logs create-log-group \
  --log-group-name /ecs/options-agent \
  --region us-east-1

aws logs put-resource-policy \
  --log-group-name /ecs/options-agent \
  --policy-name ecsOptionsAgentPolicy \
  --policy-document '{
    "Statement": [{
      "Sid": "ecslogs",
      "Effect": "Allow",
      "Principal": {"Service":"ecs-tasks.amazonaws.com"},
      "Action":"logs:PutLogEvents",
      "Resource":"arn:aws:logs:us-east-1:YOUR_ACCOUNT_ID:log-group:/ecs/options-agent:*"
    }]
  }'
```

### 1.4 Create IAM Roles

**Task Execution Role** (ECS pulls the image):

```bash
aws iam create-role \
  --role-name ecsTaskExecutionRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service":"ecs-tasks.amazonaws.com"},
      "Action":"sts:AssumeRole"
    }]
  }'

# Attach basic execution permissions
aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-name AmazonECSTaskExecutionRolePolicy \
  --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy

# Allow reading Secrets Manager
aws iam attach-role-policy \
  --role-name ecsTaskExecutionRole \
  --policy-name SecretsManagerRead \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Action": ["secretsmanager:GetSecretValue"],
      "Resource": [
        "arn:aws:secretsmanager:us-east-1:YOUR_ACCOUNT_ID:secret:options-agent:*"
      ]
    }]
  }'
```

**Task Role** (container itself — minimal permissions, no credentials needed):

```bash
aws iam create-role \
  --role-name optionsAgentTaskRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service":"ecs-tasks.amazonaws.com"},
      "Action":"sts:AssumeRole"
    }]
  }'
# This role needs no permissions — the agent only makes outbound HTTPS calls
```

### 1.5 Register Task Definition

Replace `YOUR_ACCOUNT_ID` in `aws/task-definition.json`, then:

```bash
aws ecs register-task-definition \
  --cli-input-json file://aws/task-definition.json \
  --region us-east-1
```

---

## Step 2 — Build & Push Docker Image

```bash
# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com

# Build
docker build -t options-agent .

# Tag and push
docker tag options-agent:latest \
  YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/options-agent:latest

docker push \
  YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/options-agent:latest
```

---

## Step 3 — Create ECS Cluster & Service

### Option A: Fargate (recommended — managed, no servers)

```bash
# Create cluster
aws ecs create-cluster \
  --cluster-name options-agent-cluster \
  --region us-east-1 \
  --settings name=containerInsights,value=enabled

# Create service (one long-running task, auto-restart on failure)
aws ecs create-service \
  --cluster options-agent-cluster \
  --service-name options-agent-service \
  --task-definition options-agent:1 \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={
    subnets=[subnet-XXXXXXXX,subnet-YYYYYYYY],
    securityGroups=[sg-XXXXXXXX],
    assignPublicIp=DISABLED
  }" \
  --region us-east-1
```

### Option B: EC2 t3.micro (cheaper but you manage the server)

```bash
# Create cluster
aws ecs create-cluster \
  --cluster-name options-agent-cluster \
  --region us-east-1

# Launch EC2 instance with ECS agent (use the ECS-optimized AMI)
aws ec2 run-instances \
  --image-id ami-0abcdef1234567890 \
  --instance-type t3.micro \
  --iam-instance-profile Name=ecsInstanceRole \
  --key-name your-key-pair \
  --user-data "#!/bin/bash echo 'ECS_CLUSTER=options-agent-cluster' >> /etc/ecs/ecs.config" \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=options-agent-host}]'

# Then register task definition and create service with --launch-type EC2
```

---

## Step 4 — Verify Deployment

```bash
# Check service status
aws ecs describe-services \
  --cluster options-agent-cluster \
  --services options-agent-service \
  --region us-east-1

# Check logs
aws logs tail /ecs/options-agent --follow --region us-east-1

# Check task
aws ecs list-tasks \
  --cluster options-agent-cluster \
  --region us-east-1
```

Expected log output:
```
Options Monitoring & Macro Arbitrage Agent — BOOTING
  Tickers:   ['SPY', 'QQQ']
  Interval:  30s
  Dry-run:   False
  Mock data: False
```

---

## Step 5 — Configure Alert Webhooks (Production)

Set `DRY_RUN=false` in `task-definition.json` to enable real alerts.

```bash
# Generic webhook (optional)
aws secretsmanager create-secret \
  --name options-agent/alert_webhook \
  --secret-string '{"ALERT_WEBHOOK_URL":"https://your-webhook-endpoint.com/alerts"}' \
  --region us-east-1

# Slack incoming webhook
aws secretsmanager create-secret \
  --name options-agent/slack_webhook \
  --secret-string '{"SLACK_WEBHOOK_URL":"https://hooks.slack.com/services/XXX/YYY/ZZZ"}' \
  --region us-east-1
```

Add these secrets to the container definition's `secrets` array in `task-definition.json`, then update the service:

```bash
aws ecs update-service \
  --cluster options-agent-cluster \
  --service options-agent-service \
  --task-definition options-agent:2 \
  --force-new-deployment \
  --region us-east-1
```

---

## Monitoring & Alerts

### CloudWatch Dashboard

```bash
# Create dashboard
aws cloudwatch put-dashboard \
  --dashboard-name options-agent \
  --dashboard-body '{
    "widgets": [{
      "type": "log",
      "properties": {
        "title": "Options Agent Logs",
        "region": "us-east-1",
        "logGroup": "/ecs/options-agent",
        "query": "fields @timestamp, @message | filter @message like /ALERT|HB|error/i | sort @timestamp desc | limit 50"
      }
    }]
  }'
```

### CloudWatch Alarm (restart on crash)

```bash
aws cloudwatch put-metric-alarm \
  --alarm-name options-agent-down \
  --alarm-description "Options Agent container stopped" \
  --metric-name ECSTaskCount \
  --namespace AWS/ECS \
  --statistic Average \
  --period 60 \
  --evaluation-periods 2 \
  --threshold 1 \
  --comparison-operator LessThanThreshold \
  --dimensions Name=ClusterName,Value=options-agent-cluster Name=ServiceName,Value=options-agent-service \
  --alarm-actions arn:aws:sns:us-east-1:YOUR_ACCOUNT_ID:your-sns-topic
```

---

## Updating the Agent

```bash
# 1. Pull latest code
git pull origin main

# 2. Rebuild and push image
docker build -t options-agent .
docker tag options-agent:latest \
  YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/options-agent:latest
docker push \
  YOUR_AWS_ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/options-agent:latest

# 3. Force new deployment (ECS pulls new image)
aws ecs update-service \
  --cluster options-agent-cluster \
  --service options-agent-service \
  --force-new-deployment \
  --region us-east-1
```

---

## Cost Breakdown

| Resource | Spec | Monthly Cost |
|----------|------|-------------|
| ECS Fargate (0.5 vCPU / 1GB) | 24×30=720h | ~$12 |
| CloudWatch Logs | ~10MB/day | ~$1 |
| Secrets Manager | 3 secrets | ~$3 |
| Data transfer | ~1GB/month | ~$1 |
| **Fargate Total** | | **~$17** |

| Resource | Spec | Monthly Cost |
|----------|------|-------------|
| EC2 t3.micro (on-demand) | 720h | ~$8.50 |
| EBS 30GB gp3 | storage | ~$3 |
| Data transfer | ~1GB/month | ~$1 |
| **EC2 Total** | | **~$13** |

EC2 with 1-year Reserved Instance: **~$6/month**

---

## Troubleshooting

**Container won't start:**
```bash
aws ecs describe-tasks \
  --cluster options-agent-cluster \
  --tasks <task-arn>
# Check lastStatus, stopCode, stopReason

# View logs
aws logs tail /ecs/options-agent --region us-east-1
```

**Rate limit errors from Alpaca:**
- Check `risk_thresholds.max_consecutive_errors` — increase from 5
- Alpaca Free Tier limit is 15 req/s; this should be fine for 2 tickers at 30s intervals (~4 req/cycle)

**IV Rank shows 0.5 always:**
- IV Rank needs 5+ historical data points to compute. Let it run for a few days.
- Polygon.io Free Tier provides historical IV data; ensure `POLYGON_API_KEY` is set and not empty.

---

## Data Source Costs

| Source | Tier | Cost |
|--------|------|------|
| Alpaca Market Data | Free | Free |
| Polygon.io | Starter | Free (5 keys, rate-limited) |
| Polymarket | Public API | Free |
| Kalshi | Public API | Free |

**Total data cost: $0/month** (with free tiers)
