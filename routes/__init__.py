# routes/__init__.py
from flask import Blueprint
# Import các routes
from .train_routes import train_routes
from .predic_routes import predic_routes
from .train_mongod_routes import train_mongod_routes
from .predic_mongod_routes import predic_mongod_routes
# Tạo blueprint chính
def init_app(app):
    app.register_blueprint(predic_mongod_routes)
    app.register_blueprint(train_mongod_routes)
    app.register_blueprint(train_routes)
    app.register_blueprint(predic_routes)