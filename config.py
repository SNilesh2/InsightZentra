import os

class Config:
    SQLALCHEMY_DATABASE_URI = "mysql+mysqlconnector://root:@localhost/article_summarizer"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SECRET_KEY = os.urandom(24)
