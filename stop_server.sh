#!/bin/bash

# ================================================================================
# ADAES - Stop Server Script
# ================================================================================

echo "🛑 Stopping ADAES System..."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Kill processes by PID files
if [ -f "backend.pid" ]; then
    BACKEND_PID=$(cat backend.pid)
    echo -e "${YELLOW}🔄 Stopping backend (PID: $BACKEND_PID)...${NC}"
    kill $BACKEND_PID 2>/dev/null
    rm -f backend.pid
fi

if [ -f "frontend.pid" ]; then
    FRONTEND_PID=$(cat frontend.pid)
    echo -e "${YELLOW}🔄 Stopping frontend (PID: $FRONTEND_PID)...${NC}"
    kill $FRONTEND_PID 2>/dev/null
    rm -f frontend.pid
fi

# Kill all Python processes
echo -e "${YELLOW}🐍 Killing all Python processes...${NC}"
pkill -f python 2>/dev/null || true
pkill -f python3 2>/dev/null || true

# Kill processes on specific ports
echo -e "${YELLOW}🔍 Killing processes on ports 5001 and 3000...${NC}"
lsof -ti:5001 | xargs kill -9 2>/dev/null || true
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

echo -e "${GREEN}✅ ADAES system stopped!${NC}"

