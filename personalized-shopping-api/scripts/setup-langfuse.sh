#!/bin/bash
set -e

echo "=========================================="
echo "LangFuse Setup Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}This script will help you set up LangFuse for LLM observability${NC}"
echo ""

# Check if docker-compose is running
if ! docker ps | grep -q langfuse; then
    echo -e "${YELLOW}Starting LangFuse services...${NC}"
    docker-compose up -d langfuse-db langfuse
    echo ""
    echo -e "${YELLOW}Waiting for services to be healthy (this may take 30-60 seconds)...${NC}"
    sleep 10

    # Wait for langfuse to be healthy
    for i in {1..30}; do
        if docker ps --filter "name=langfuse" --filter "health=healthy" | grep -q langfuse; then
            echo -e "${GREEN}✓ LangFuse is healthy!${NC}"
            break
        fi
        echo -n "."
        sleep 2
    done
    echo ""
fi

echo ""
echo -e "${GREEN}=========================================="
echo "LangFuse is ready!"
echo "==========================================${NC}"
echo ""
echo -e "1. Open LangFuse UI: ${GREEN}http://localhost:3002${NC}"
echo ""
echo -e "2. ${YELLOW}Create an account${NC} (first time only):"
echo "   - Click 'Sign Up'"
echo "   - Use any email (e.g., admin@localhost)"
echo "   - Set a password"
echo ""
echo -e "3. ${YELLOW}Create a project${NC}:"
echo "   - After logging in, create a new project"
echo "   - Name it 'shopping-assistant'"
echo ""
echo -e "4. ${YELLOW}Get API Keys${NC}:"
echo "   - Go to Project Settings"
echo "   - Navigate to 'API Keys' tab"
echo "   - Click 'Create new API keys'"
echo "   - Copy the Public Key and Secret Key"
echo ""
echo -e "5. ${YELLOW}Update your .env file${NC}:"
echo "   Add the following lines to your .env file:"
echo ""
echo "   LANGFUSE_ENABLED=true"
echo "   LANGFUSE_PUBLIC_KEY=pk-lf-..."
echo "   LANGFUSE_SECRET_KEY=sk-lf-..."
echo "   LANGFUSE_HOST=http://langfuse:3000"
echo ""
echo -e "6. ${YELLOW}Restart the API${NC}:"
echo "   docker-compose restart api"
echo ""
echo -e "${GREEN}Then test your setup by making an API call!${NC}"
echo ""
echo "=========================================="
