import os
from flask import Flask
from routes.api import api_bp
from config import get_config

def create_app(config=None):
    """Application factory function"""
    app = Flask(__name__)
    
    # Load configuration
    app_config = config if config else get_config()
    app.config.from_object(app_config)
    
    # Register blueprints
    app.register_blueprint(api_bp)
    
    return app

if __name__ == '__main__':
    # Create app instance
    app = create_app()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)