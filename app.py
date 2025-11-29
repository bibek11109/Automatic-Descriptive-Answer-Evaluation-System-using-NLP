"""
ADAES - Automatic Descriptive Answer Evaluation System
Flask Backend Application
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
from models import db, User, Question, Submission
from simple_scorer import get_scorer
from improved_model_loader import get_improved_model_instance
from config import Config
import os

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize extensions
CORS(app, 
     origins=['http://localhost:3000', 'http://localhost:5173', 'http://localhost:5001'],
     allow_headers=['Content-Type', 'Authorization'],
     methods=['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'])
db.init_app(app)
jwt = JWTManager(app)

# Initialize AI model (with fallback to simple scorer)
def initialize_ai_model():
    global ai_model, use_ai_model
    model_path = os.path.join('..', 'models', 'best_model.pt')
    print(f"🔍 Looking for model at: {os.path.abspath(model_path)}")

    if os.path.exists(model_path):
        print("🤖 Loading AI model...")
        try:
            ai_model = get_improved_model_instance(
                model_path=model_path,
                device='cpu'
            )
            print("✅ AI model loaded successfully! (95% accuracy)")
            use_ai_model = True
        except Exception as e:
            print(f"❌ Failed to load AI model: {e}")
            import traceback
            traceback.print_exc()
            print("Using simple scorer as fallback")
            ai_model = None
            use_ai_model = False
    else:
        print("⚠️ AI model not found, using simple scorer")
        ai_model = None
        use_ai_model = False

# Initialize the model
initialize_ai_model()

# Initialize simple scorer as fallback
simple_scorer = get_scorer()
print("✅ Simple scorer ready (keyword-based)")

# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

@app.cli.command()
def init_db():
    """Initialize database"""
    with app.app_context():
        db.create_all()
        print("✅ Database initialized!")

# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    """Register new user"""
    data = request.json
    
    # Validate input
    if not all(k in data for k in ['email', 'password', 'name', 'role']):
        return jsonify({'error': 'Missing required fields'}), 400
    
    if data['role'] not in ['teacher', 'student']:
        return jsonify({'error': 'Invalid role'}), 400
    
    # Check if user exists
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 400
    
    # Create user
    user = User(
        email=data['email'],
        name=data['name'],
        role=data['role']
    )
    user.set_password(data['password'])
    
    db.session.add(user)
    db.session.commit()
    
    # Create token
    token = create_access_token(identity=str(user.id))
    
    return jsonify({
        'message': 'User registered successfully',
        'token': token,
        'user': user.to_dict()
    }), 201

@app.route('/api/auth/login', methods=['POST'])
def login():
    """Login user"""
    data = request.json
    
    if not all(k in data for k in ['email', 'password']):
        return jsonify({'error': 'Missing email or password'}), 400
    
    user = User.query.filter_by(email=data['email']).first()
    
    if not user or not user.check_password(data['password']):
        return jsonify({'error': 'Invalid credentials'}), 401
    
    token = create_access_token(identity=str(user.id))
    
    return jsonify({
        'message': 'Login successful',
        'token': token,
        'user': user.to_dict()
    })

@app.route('/api/auth/me', methods=['GET'])
@jwt_required()
def get_current_user():
    """Get current user info"""
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify(user.to_dict())

# ============================================================================
# QUESTION ROUTES (Teacher)
# ============================================================================

@app.route('/api/questions', methods=['GET'])
@jwt_required()
def get_questions():
    """Get all questions (teacher: own questions, student: all questions)"""
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    
    if user.role == 'teacher':
        questions = Question.query.filter_by(teacher_id=user_id).all()
    else:
        questions = Question.query.all()
    
    return jsonify([q.to_dict() for q in questions])

@app.route('/api/questions/<int:question_id>', methods=['GET'])
@jwt_required()
def get_question(question_id):
    """Get specific question"""
    question = Question.query.get(question_id)
    
    if not question:
        return jsonify({'error': 'Question not found'}), 404
    
    return jsonify(question.to_dict())

@app.route('/api/questions', methods=['POST'])
@jwt_required()
def create_question():
    """Create new question (teachers only)"""
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    
    if user.role != 'teacher':
        return jsonify({'error': 'Only teachers can create questions'}), 403
    
    data = request.json
    
    # Handle both frontend field names (title/description) and backend field names (subject/question_text)
    title = data.get('title') or data.get('subject', 'General')
    description = data.get('description') or data.get('question_text', '')
    reference_answer = data.get('reference_answer', '')
    
    if not description or not reference_answer:
        return jsonify({'error': 'Missing required fields: description and reference_answer'}), 400
    
    question = Question(
        teacher_id=user_id,
        question_text=description,
        reference_answer=reference_answer,
        subject=title,
        max_score=data.get('max_score', 60)
    )
    
    db.session.add(question)
    db.session.commit()
    
    return jsonify({
        'message': 'Question created successfully',
        'question': question.to_dict()
    }), 201

@app.route('/api/questions/<int:question_id>', methods=['PUT'])
@jwt_required()
def update_question(question_id):
    """Update question (teacher who created it only)"""
    user_id = int(get_jwt_identity())
    question = Question.query.get(question_id)
    
    if not question:
        return jsonify({'error': 'Question not found'}), 404
    
    if question.teacher_id != user_id:
        return jsonify({'error': 'Not authorized'}), 403
    
    data = request.json
    
    if 'question_text' in data:
        question.question_text = data['question_text']
    if 'reference_answer' in data:
        question.reference_answer = data['reference_answer']
    if 'subject' in data:
        question.subject = data['subject']
    if 'max_score' in data:
        question.max_score = data['max_score']
    
    db.session.commit()
    
    return jsonify({
        'message': 'Question updated successfully',
        'question': question.to_dict()
    })

@app.route('/api/questions/<int:question_id>', methods=['DELETE'])
@jwt_required()
def delete_question(question_id):
    """Delete question (teacher who created it only)"""
    user_id = int(get_jwt_identity())
    question = Question.query.get(question_id)
    
    if not question:
        return jsonify({'error': 'Question not found'}), 404
    
    if question.teacher_id != user_id:
        return jsonify({'error': 'Not authorized'}), 403
    
    db.session.delete(question)
    db.session.commit()
    
    return jsonify({'message': 'Question deleted successfully'})

@app.route('/api/questions/<int:question_id>/toggle-visibility', methods=['POST'])
@jwt_required()
def toggle_result_visibility(question_id):
    """Toggle result visibility for students (teacher only)"""
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    
    if user.role != 'teacher':
        return jsonify({'error': 'Only teachers can toggle result visibility'}), 403
    
    question = Question.query.get(question_id)
    if not question:
        return jsonify({'error': 'Question not found'}), 404
    
    if question.teacher_id != int(user_id):
        return jsonify({'error': 'Not authorized'}), 403
    
    # Toggle visibility
    question.result_visibility = not question.result_visibility
    db.session.commit()
    
    return jsonify({
        'message': f'Result visibility {"enabled" if question.result_visibility else "disabled"} for students',
        'result_visibility': question.result_visibility
    })

# ============================================================================
# SUBMISSION ROUTES
# ============================================================================

@app.route('/api/submissions', methods=['POST'])
@jwt_required()
def submit_answer():
    """Submit answer for evaluation"""
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    
    if user.role != 'student':
        return jsonify({'error': 'Only students can submit answers'}), 403
    
    data = request.json
    
    if not all(k in data for k in ['question_id', 'student_answer']):
        return jsonify({'error': 'Missing required fields'}), 400
    
    question = Question.query.get(data['question_id'])
    if not question:
        return jsonify({'error': 'Question not found'}), 404
    
    # Evaluate using AI model or simple scorer
    try:
        if use_ai_model:
            evaluation = ai_model.evaluate_answer(
                data['student_answer'],
                reference_answer=question.reference_answer,
                max_score=question.max_score,
                subject=question.subject
            )
        else:
            evaluation = simple_scorer.evaluate_answer(
                data['student_answer'],
                reference_answer=question.reference_answer,
                max_score=question.max_score
            )
    except Exception as e:
        return jsonify({'error': f'Evaluation failed: {str(e)}'}), 500
    
    # Extract scores from new comprehensive feedback format
    final_score = evaluation.get('score', evaluation.get('final_score', 0))
    normalized_score = evaluation.get('percentage', 0) / 100.0
    feedback_text = evaluation.get('overall_feedback', evaluation.get('feedback', ''))
    
    # Check if this is an edit
    if data.get('is_edit') and data.get('original_submission_id'):
        # Update existing submission
        original_submission = Submission.query.get(data['original_submission_id'])
        if not original_submission:
            return jsonify({'error': 'Original submission not found'}), 404
        
        if original_submission.student_id != user_id:
            return jsonify({'error': 'Not authorized'}), 403
        
        if original_submission.edit_count >= 3:
            return jsonify({'error': 'Maximum edit attempts reached'}), 400
        
        # Update the submission
        original_submission.student_answer = data['student_answer']
        original_submission.final_score = final_score
        original_submission.normalized_score = normalized_score
        original_submission.feedback = feedback_text
        original_submission.edit_count += 1
        
        db.session.commit()
        
        return jsonify({
            'message': 'Answer updated successfully',
            'submission': original_submission.to_dict(include_answer=True),
            'evaluation': evaluation
        })
    else:
        # Create new submission
        submission = Submission(
            question_id=data['question_id'],
            student_id=user_id,
            student_answer=data['student_answer'],
            final_score=final_score,
            normalized_score=normalized_score,
            feedback=feedback_text
        )
        
        db.session.add(submission)
        db.session.commit()
        
        return jsonify({
            'message': 'Answer submitted successfully',
            'submission': submission.to_dict(include_answer=True),
            'evaluation': evaluation
        })

@app.route('/api/submissions/my', methods=['GET'])
@jwt_required()
def get_my_submissions():
    """Get current user's submissions"""
    user_id = int(get_jwt_identity())
    submissions = Submission.query.filter_by(student_id=user_id).all()
    
    return jsonify([s.to_dict(include_answer=True) for s in submissions])

@app.route('/api/submissions/all', methods=['GET'])
@jwt_required()
def get_all_submissions():
    """Get all submissions for teachers (latest version only)"""
    user_id = int(get_jwt_identity())
    
    # Get all questions created by this teacher
    teacher_questions = Question.query.filter_by(teacher_id=user_id).all()
    question_ids = [q.id for q in teacher_questions]
    
    if not question_ids:
        return jsonify([])
    
    # Get all submissions for these questions
    all_submissions = Submission.query.filter(Submission.question_id.in_(question_ids)).all()
    
    # Group by question_id and student_id to get only the latest submission
    latest_submissions = {}
    for submission in all_submissions:
        key = (submission.question_id, submission.student_id)
        if key not in latest_submissions or submission.submitted_at > latest_submissions[key].submitted_at:
            latest_submissions[key] = submission
    
    return jsonify([s.to_dict(include_answer=True) for s in latest_submissions.values()])

@app.route('/api/submissions/<int:submission_id>/update', methods=['PUT'])
@jwt_required()
def update_submission(submission_id):
    """Update submission feedback and marks"""
    user_id = int(get_jwt_identity())
    submission = Submission.query.get_or_404(submission_id)
    
    # Check if user is the teacher who created the question
    question = submission.question
    if question.teacher_id != user_id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    new_feedback = data.get('feedback', '')
    new_score = data.get('final_score', None)
    
    if new_feedback:
        submission.feedback = new_feedback
    
    if new_score is not None:
        submission.final_score = float(new_score)
        # Recalculate normalized score
        submission.normalized_score = submission.final_score / question.max_score
    
    db.session.commit()
    
    return jsonify({
        'message': 'Submission updated successfully',
        'submission': submission.to_dict(include_answer=True)
    })

@app.route('/api/submissions/<int:submission_id>', methods=['GET'])
@jwt_required()
def get_submission(submission_id):
    """Get specific submission details"""
    user_id = int(get_jwt_identity())
    submission = Submission.query.get(submission_id)
    
    if not submission:
        return jsonify({'error': 'Submission not found'}), 404
    
    # Check authorization
    if submission.student_id != user_id and submission.question.teacher_id != user_id:
        return jsonify({'error': 'Not authorized'}), 403
    
    return jsonify(submission.to_dict(include_answer=True))

@app.route('/api/questions/<int:question_id>/submissions', methods=['GET'])
@jwt_required()
def get_question_submissions(question_id):
    """Get all submissions for a question (teacher only)"""
    user_id = int(get_jwt_identity())
    question = Question.query.get(question_id)
    
    if not question:
        return jsonify({'error': 'Question not found'}), 404
    
    if question.teacher_id != user_id:
        return jsonify({'error': 'Not authorized'}), 403
    
    submissions = Submission.query.filter_by(question_id=question_id).all()
    
    return jsonify([s.to_dict(include_answer=True) for s in submissions])

# ============================================================================
# EVALUATION ENDPOINT (Public for testing)
# ============================================================================

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Quick evaluation without saving (for testing)"""
    data = request.json
    
    if 'student_answer' not in data:
        return jsonify({'error': 'Missing student_answer'}), 400
    
    max_score = data.get('max_score', 60)
    reference_answer = data.get('reference_answer', '')
    subject = data.get('subject', '')
    
    try:
        if use_ai_model:
            evaluation = ai_model.evaluate_answer(
                data['student_answer'],
                reference_answer=reference_answer,
                max_score=max_score,
                subject=subject
            )
        else:
            evaluation = simple_scorer.evaluate_answer(
                data['student_answer'],
                reference_answer=reference_answer,
                max_score=max_score
            )
        return jsonify(evaluation)
    except Exception as e:
        return jsonify({'error': f'Evaluation failed: {str(e)}'}), 500

# ============================================================================
# DASHBOARD STATS
# ============================================================================

@app.route('/api/stats/teacher', methods=['GET'])
@jwt_required()
def teacher_stats():
    """Get teacher dashboard statistics"""
    user_id = int(get_jwt_identity())
    user = User.query.get(user_id)
    
    if user.role != 'teacher':
        return jsonify({'error': 'Teachers only'}), 403
    
    questions_count = Question.query.filter_by(teacher_id=user_id).count()
    submissions_count = db.session.query(Submission).join(Question).filter(
        Question.teacher_id == user_id
    ).count()
    
    # Get unique students who have submitted
    active_students = db.session.query(Submission.student_id).join(Question).filter(
        Question.teacher_id == user_id
    ).distinct().count()
    
    # Get recent submissions with student names
    recent_submissions = db.session.query(Submission, User.name, Question.question_text, Question.subject).join(
        User, Submission.student_id == User.id
    ).join(
        Question, Submission.question_id == Question.id
    ).filter(
        Question.teacher_id == user_id
    ).order_by(Submission.submitted_at.desc()).limit(5).all()
    
    # Calculate average score
    avg_score = db.session.query(db.func.avg(Submission.final_score)).join(Question).filter(
        Question.teacher_id == user_id
    ).scalar() or 0
    
    recent_activity = []
    for submission, student_name, question_text, subject in recent_submissions:
        recent_activity.append({
            'student_name': student_name,
            'question_text': question_text[:100] + '...' if len(question_text) > 100 else question_text,
            'subject': subject,
            'submitted_at': submission.submitted_at.isoformat(),
            'score': submission.final_score
        })
    
    return jsonify({
        'questions_count': questions_count,
        'submissions_count': submissions_count,
        'active_students': active_students,
        'average_score': round(avg_score, 2),
        'recent_activity': recent_activity
    })

@app.route('/api/stats/student', methods=['GET'])
@jwt_required()
def student_stats():
    """Get student dashboard statistics"""
    user_id = int(get_jwt_identity())
    
    submissions = Submission.query.filter_by(student_id=user_id).all()
    
    # Calculate average score
    avg_score = sum(s.final_score for s in submissions) / len(submissions) if submissions else 0
    
    # Calculate recent submissions (this week)
    from datetime import datetime, timedelta
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_submissions = len([s for s in submissions if s.submitted_at >= week_ago])
    
    # Calculate best score
    best_score = max(s.final_score for s in submissions) if submissions else 0
    
    # Check if any results are visible to the student
    has_visible_results = any(
        submission.question.result_visibility for submission in submissions
    ) if submissions else False
    
    return jsonify({
        'submissions_count': len(submissions),
        'average_score': round(avg_score, 2),
        'recent_submissions': recent_submissions,
        'best_score': round(best_score, 2),
        'has_visible_results': has_visible_results
    })

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        print("✅ Database initialized!")
        initialize_ai_model()
    
    print("\n" + "="*60)
    print("🚀 ADAES Backend Server Starting...")
    print("="*60)
    print(f"📍 Server: http://localhost:5001")
    print(f"📊 Database: {app.config['SQLALCHEMY_DATABASE_URI']}")
    print(f"🤖 AI Model: {'Loaded' if ai_model else 'Not loaded'}")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5001)

