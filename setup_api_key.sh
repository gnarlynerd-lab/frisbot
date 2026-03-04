#!/bin/bash

# Script to help set up DeepSeek API key

echo "🔑 Frisbot API Key Setup"
echo "========================"
echo ""
echo "To use the chat feature, you need a DeepSeek API key."
echo ""
echo "Steps to get your API key:"
echo "1. Visit https://platform.deepseek.com/"
echo "2. Sign up or log in"
echo "3. Go to API Keys section"
echo "4. Create a new API key"
echo "5. Copy the key"
echo ""
echo "DeepSeek pricing (as of 2024):"
echo "  • $0.14 per million input tokens"
echo "  • $0.28 per million output tokens"
echo "  • ~100x cheaper than GPT-4"
echo ""
read -p "Do you have your DeepSeek API key ready? (y/n): " has_key

if [[ "$has_key" == "y" ]]; then
    echo ""
    echo "Please enter your DeepSeek API key:"
    read -s api_key
    echo ""
    
    if [[ -n "$api_key" ]]; then
        # Backup existing .env if it exists
        if [[ -f .env ]]; then
            cp .env .env.backup
            echo "✓ Backed up existing .env to .env.backup"
        fi
        
        # Update the .env file
        cat > .env << EOF
# DeepSeek API Configuration
DEEPSEEK_API_KEY=$api_key

# Optional: Database path (defaults to frisbot.db)
# DATABASE_PATH=frisbot.db

# Optional: CORS origins (comma-separated)
# CORS_ORIGINS=http://localhost:3000,http://localhost:3001
EOF
        
        echo "✅ API key configured successfully!"
        echo ""
        echo "The servers will automatically reload with the new configuration."
        echo "You can now use the chat feature at http://localhost:3001"
        
        # Test the API key
        echo ""
        echo "Testing API key..."
        python -c "
import os
from dotenv import load_dotenv
import requests

load_dotenv()
api_key = os.getenv('DEEPSEEK_API_KEY')
headers = {'Authorization': f'Bearer {api_key}'}
try:
    # Simple test request
    response = requests.get('https://api.deepseek.com/v1/models', headers=headers, timeout=5)
    if response.status_code == 200:
        print('✅ API key is valid!')
    else:
        print('⚠️  API key might be invalid. Status:', response.status_code)
except Exception as e:
    print('⚠️  Could not verify API key:', str(e))
    print('    This might be a network issue. Try using the chat to test.')
" 2>/dev/null || echo "Note: Could not verify API key automatically. Try using the chat to test."
        
    else
        echo "❌ No API key entered. Setup cancelled."
    fi
else
    echo ""
    echo "To set up later, run this script again or edit .env directly."
    echo "Replace 'your_deepseek_api_key_here' with your actual API key."
fi