from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Paper(db.Model):
    __tablename__ = "papers"

    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.Text, nullable=False)
    authors = db.Column(db.Text, nullable=False)
    link = db.Column(db.Text, nullable=False, unique=True)
    pdf_link = db.Column(db.Text, nullable=False)
    match_type = db.Column(db.Text, nullable=False)
    matched_terms = db.Column(db.Text, nullable=False)
    publication_date = db.Column(db.Text)
    scraped_date = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())
