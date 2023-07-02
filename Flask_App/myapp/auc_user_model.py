from flask_login import UserMixin
from . import db

class User(UserMixin, db.Model):
    """
    Create a User object.
    Args:
        id: primary key required by SQLAlchemy
        email: Email of the user. must be Unique
        password: User password
        name: User name
    """
    id = db.Column(db.Integer, primary_key=True) # primary keys are required by SQLAlchemy
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))
    name = db.Column(db.String(1000))

    def __repr__(self):
        return f"NAme = {self.name} | ID = {self.id} | email = {self.email}"