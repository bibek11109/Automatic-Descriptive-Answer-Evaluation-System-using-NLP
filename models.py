"""
Database Models for ADAES
"""

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

db = SQLAlchemy()


class User(db.Model):
    """User model for teachers and students"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'teacher' or 'student'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    questions = db.relationship('Question', backref='teacher', lazy=True, cascade='all, delete-orphan')
    submissions = db.relationship('Submission', backref='student', lazy=True, cascade='all, delete-orphan')
    
    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Verify password"""
        return check_password_hash(self.password_hash, password)
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'email': self.email,
            'name': self.name,
            'role': self.role,
            'created_at': self.created_at.isoformat()
        }


class Question(db.Model):
    """Question model with reference answers"""
    __tablename__ = 'questions'
    
    id = db.Column(db.Integer, primary_key=True)
    teacher_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    question_text = db.Column(db.Text, nullable=False)
    reference_answer = db.Column(db.Text, nullable=False)
    subject = db.Column(db.String(100))
    max_score = db.Column(db.Integer, default=60)
    result_visibility = db.Column(db.Boolean, default=False)  # False = hidden, True = visible to students
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    submissions = db.relationship('Submission', backref='question', lazy=True, cascade='all, delete-orphan')
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'teacher_id': self.teacher_id,
            'teacher_name': self.teacher.name if self.teacher else None,
            'question_text': self.question_text,
            'reference_answer': self.reference_answer,
            'subject': self.subject,
            'max_score': self.max_score,
            'result_visibility': self.result_visibility,
            'created_at': self.created_at.isoformat(),
            'submission_count': len(self.submissions)
        }


class Submission(db.Model):
    """Student submission model"""
    __tablename__ = 'submissions'
    
    id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('questions.id'), nullable=False)
    student_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    student_answer = db.Column(db.Text, nullable=False)
    
    # Scores
    final_score = db.Column(db.Float, nullable=False)
    normalized_score = db.Column(db.Float)
    
    # Feedback
    feedback = db.Column(db.Text)
    
    # Metadata
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    edit_count = db.Column(db.Integer, default=0)
    
    def to_dict(self, include_answer=False):
        """Convert to dictionary"""
        data = {
            'id': self.id,
            'question_id': self.question_id,
            'student_id': self.student_id,
            'student_name': self.student.name if self.student else None,
            'final_score': round(self.final_score, 2),
            'normalized_score': round(self.normalized_score, 4) if self.normalized_score else None,
            'percentage': f"{self.normalized_score * 100:.1f}%" if self.normalized_score else None,
            'feedback': self.feedback,
            'submitted_at': self.submitted_at.isoformat(),
            'edit_count': self.edit_count,
            'student_answer': self.student_answer,
            'question': {
                'id': self.question.id if self.question else None,
                'question_text': self.question.question_text if self.question else None,
                'subject': self.question.subject if self.question else None,
                'max_score': self.question.max_score if self.question else None,
                'result_visibility': self.question.result_visibility if self.question else None
            } if self.question else None
        }
        
        return data

