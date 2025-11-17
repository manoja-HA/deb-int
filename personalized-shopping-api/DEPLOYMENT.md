# Deployment Guide

Complete guide for deploying the Personalized Shopping Assistant API.

## 🚀 Docker Compose Deployment (Recommended)

### Step 1: Clone and Navigate

```bash
cd personalized-shopping-api
```

### Step 2: Configure Environment

```bash
# Copy example environment file
cp .env .env.local

# Edit as needed
nano .env
```

### Step 3: Start Services

```bash
# Quick start (recommended)
./quickstart.sh

# OR manual start
docker-compose up -d
```

### Step 4: Verify Services

```bash
# Check all services
docker-compose ps

# View logs
docker-compose logs -f api

# Test health endpoint
curl http://localhost:8000/health
```

### Step 5: Test the API

```bash
# Run test script
./test_api.sh

# OR manual test
curl -X POST http://localhost:8000/api/v1/recommendations/personalized \
  -H 'Content-Type: application/json' \
  -d '{"query": "What would Kenneth Martinez like?", "customer_name": "Kenneth Martinez"}'
```

## 📦 Production Deployment

### Using Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml shopping-api

# Check services
docker service ls

# View logs
docker service logs -f shopping-api_api
```

### Using Kubernetes

```bash
# Convert docker-compose to k8s
kompose convert -f docker-compose.yml

# Deploy
kubectl apply -f .

# Check pods
kubectl get pods

# View logs
kubectl logs -f deployment/api
```

## 🔒 Production Checklist

### Security

- [ ] Change `SECRET_KEY` in `.env`
- [ ] Set strong passwords for Grafana
- [ ] Enable HTTPS/TLS
- [ ] Configure firewall rules
- [ ] Set up authentication/authorization
- [ ] Review CORS settings

### Performance

- [ ] Increase worker count: `WORKERS=4`
- [ ] Enable Redis caching: `ENABLE_CACHE=true`
- [ ] Pre-build vector index
- [ ] Configure resource limits
- [ ] Set up load balancing

### Monitoring

- [ ] Configure Prometheus retention
- [ ] Set up Grafana dashboards
- [ ] Enable error tracking
- [ ] Configure log aggregation
- [ ] Set up alerting

### Data

- [ ] Add production customer data
- [ ] Build vector index with full dataset
- [ ] Set up data backup strategy
- [ ] Configure data retention policies

## 🔧 Configuration

### Environment Variables

```bash
# Application
ENVIRONMENT=production
DEBUG=false

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=4

# Security
SECRET_KEY=your-secret-key-here

# LLM
OLLAMA_BASE_URL=http://ollama:11434
PROFILING_MODEL=llama3.2:3b
RESPONSE_MODEL=llama3.1:8b

# Cache
REDIS_URL=redis://redis:6379
ENABLE_CACHE=true
CACHE_TTL_SECONDS=3600

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO
```

### Resource Limits

Update `docker-compose.yml`:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

## 📊 Scaling

### Horizontal Scaling

```bash
# Scale API service
docker-compose up -d --scale api=3

# With Docker Swarm
docker service scale shopping-api_api=3
```

### Load Balancing

Add nginx as reverse proxy:

```yaml
# docker-compose.yml
nginx:
  image: nginx:alpine
  ports:
    - "80:80"
  volumes:
    - ./nginx.conf:/etc/nginx/nginx.conf
  depends_on:
    - api
```

## 🔄 Updates

### Update Application Code

```bash
# Pull latest changes
git pull

# Rebuild and restart
docker-compose build api
docker-compose up -d api

# OR using quick restart
docker-compose restart api
```

### Update Models

```bash
# Enter Ollama container
docker exec -it shopping-ollama ollama pull llama3.1:8b

# Restart API
docker-compose restart api
```

### Update Vector Index

```bash
# Rebuild index
docker exec -it shopping-api python scripts/build_vector_index.py

# Restart API
docker-compose restart api
```

## 🐛 Troubleshooting

### Check Service Status

```bash
docker-compose ps
docker-compose logs api
docker-compose logs ollama
```

### Restart Services

```bash
# Restart specific service
docker-compose restart api

# Restart all
docker-compose restart
```

### Clean Start

```bash
# Stop and remove everything
docker-compose down -v

# Rebuild from scratch
docker-compose build --no-cache
docker-compose up -d
```

### View Resource Usage

```bash
docker stats
```

## 📈 Monitoring

### Access Monitoring Tools

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **API Metrics**: http://localhost:8000/metrics

### Key Metrics to Monitor

- Request rate and latency
- Agent execution times
- Cache hit rate
- Error rate
- Memory and CPU usage

### Alerting

Configure Prometheus alerting rules:

```yaml
# prometheus/alert.rules.yml
groups:
  - name: api_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(http_requests_total{status="500"}[5m]) > 0.05
        annotations:
          summary: "High error rate detected"
```

## 🔐 Security Best Practices

1. **Use Secrets Management**
   ```bash
   docker secret create api_secret_key secret.txt
   ```

2. **Enable TLS**
   - Use reverse proxy (nginx/traefik)
   - Configure SSL certificates
   - Force HTTPS redirect

3. **Network Isolation**
   ```yaml
   networks:
     internal:
       internal: true
     external:
       external: true
   ```

4. **Regular Updates**
   ```bash
   docker-compose pull
   docker-compose up -d
   ```

## 📝 Maintenance

### Backup Data

```bash
# Backup vector index
docker cp shopping-api:/app/data/embeddings ./backup/

# Backup CSV data
docker cp shopping-api:/app/data/raw ./backup/

# Backup configuration
cp .env ./backup/
```

### Log Rotation

```yaml
# docker-compose.yml
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

### Health Checks

```bash
# Automated health check
curl -f http://localhost:8000/health || exit 1
```

## 🚨 Common Issues

### Issue: API Not Starting

**Solution:**
```bash
docker-compose logs api
# Check for port conflicts, missing env vars
```

### Issue: Ollama Connection Failed

**Solution:**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Update OLLAMA_BASE_URL in .env
```

### Issue: Vector Index Not Found

**Solution:**
```bash
docker exec -it shopping-api python scripts/build_vector_index.py
```

### Issue: Out of Memory

**Solution:**
```yaml
# Reduce workers in docker-compose.yml
environment:
  WORKERS: 1
```

## 📞 Support

- **Documentation**: http://localhost:8000/api/v1/docs
- **Logs**: `docker-compose logs -f`
- **Health**: http://localhost:8000/health
- **Metrics**: http://localhost:8000/metrics

---

**Ready for Production!** 🎉

Start with: `./quickstart.sh`
