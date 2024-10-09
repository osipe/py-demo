# routes/__init__.py
from flask import Blueprint
# Import các routes
from .train_routes import train_routes
from .predic_routes import predic_routes
# Tạo blueprint chính
def init_app(app):
    app.register_blueprint(train_routes)
    app.register_blueprint(predic_routes)