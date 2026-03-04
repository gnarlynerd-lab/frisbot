#!/bin/bash

# Frisbot Full Stack Startup Script
# This script starts both backend and frontend services

echo "🚀 Starting Frisbot..."
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if port is in use
check_port() {
    lsof -i :$1 > /dev/null 2>&1
    return $?
}

# Kill existing processes on our ports if needed
if check_port 8000; then
    echo -e "${YELLOW}⚠ Port 8000 is in use. Kill existing process? (y/n)${NC}"
    read -r response
    if [[ "$response" == "y" ]]; then
        kill $(lsof -ti:8000) 2>/dev/null
        echo "✓ Killed process on port 8000"
    fi
fi

if check_port 3000; then
    echo -e "${YELLOW}⚠ Port 3000 is in use. Kill existing process? (y/n)${NC}"
    read -r response
    if [[ "$response" == "y" ]]; then
        kill $(lsof -ti:3000) 2>/dev/null
        echo "✓ Killed process on port 3000"
    fi
fi

# Check for DeepSeek API key
if grep -q "your_deepseek_api_key_here" .env 2>/dev/null; then
    echo -e "${YELLOW}⚠ WARNING: DeepSeek API key not configured in .env${NC}"
    echo "  Chat functionality will not work without an API key."
    echo "  Get one at: https://platform.deepseek.com/"
    echo ""
fi

# Start backend
echo -e "${BLUE}Starting backend server...${NC}"
source venv/bin/activate
python main.py &
BACKEND_PID=$!
echo "✓ Backend started (PID: $BACKEND_PID) on http://localhost:8000"

# Start frontend
echo -e "${BLUE}Starting frontend server...${NC}"
cd frontend
./node_modules/.bin/next dev &
FRONTEND_PID=$!
cd ..
echo "✓ Frontend starting on http://localhost:3000"

echo ""
echo -e "${GREEN}✅ Frisbot is running!${NC}"
echo ""
echo "📍 Access points:"
echo "   • Frontend: http://localhost:3000"
echo "   • Backend API: http://localhost:8000"
echo "   • API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down Frisbot..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✓ All services stopped"
    exit 0
}

# Set up signal handling
trap cleanup INT TERM

# Keep script running
wait $BACKEND_PID
wait $FRONTEND_PID