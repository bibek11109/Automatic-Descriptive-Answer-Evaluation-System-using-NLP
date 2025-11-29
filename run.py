"""
ADAES Backend - Simple Run Script
For easy startup
"""

import os
import sys

# Add parent directory to path for model imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app, db

if __name__ == '__main__':
    print("=" * 60)
    print("ADAES Backend Server")
    print("=" * 60)
    
    # Initialize database
    with app.app_context():
        db.create_all()
        print("✅ Database initialized!")
        print("✅ Simple scorer ready (keyword-based)")
    
    print("\n📍 Server: http://localhost:5001")
    print("📝 Test: POST http://localhost:5001/api/evaluate")
    print("\nPress CTRL+C to stop\n")
    
    # Run server
    app.run(debug=True, host='0.0.0.0', port=5001)

