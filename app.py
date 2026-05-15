from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os
import mysql.connector
from main import run_training_pipeline

app = Flask(__name__)
CORS(app)

# Load the best model (Decision Tree)
MODEL_PATH = 'models/dt_model.pkl'

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    model = None
    print(f"Error: Model not found at {MODEL_PATH}")

# Label Mappings (based on LabelEncoder logic)
# Features: b=0, o=1, x=2
FEATURE_MAP = {'b': 0, 'o': 1, 'x': 2}
# Target: 0=negative (Loss for X/Win for O), 1=positive (Win for X)
TARGET_MAP = {0: 'negative', 1: 'positive'}

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    board = data.get('board')
    
    if not board or len(board) != 9:
        return jsonify({'error': 'Invalid board data.'}), 400
    
    try:
        numeric_board = [FEATURE_MAP[s.lower()] for s in board]
        prediction = model.predict([numeric_board])[0]
        result = TARGET_MAP[prediction]
        return jsonify({'prediction': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def check_winner(board):
    """Helper to check if there's a winner or a draw."""
    lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8], # rows
        [0, 3, 6], [1, 4, 7], [2, 5, 8], # cols
        [0, 4, 8], [2, 4, 6]             # diagonals
    ]
    for a, b, c in lines:
        if board[a].lower() != 'b' and board[a].lower() == board[b].lower() == board[c].lower():
            return board[a].lower()
    if 'b' not in [s.lower() for s in board]:
        return 'draw'
    return None

def evaluate_board(board, ai_symbol):
    """Uses the ML model to evaluate the board state heuristic."""
    if model is None:
        return 0.5
    try:
        numeric_board = [FEATURE_MAP[s.lower()] for s in board]
        X_input = np.array([numeric_board])
        # Target for AI: 1 is 'positive' (X wins), 0 is 'negative' (O wins)
        target_idx = 1 if ai_symbol == 'x' else 0
        
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_input)[0]
            # Probabilities are usually indexed by the classes in the model
            # Assuming 0 is negative and 1 is positive based on TARGET_MAP
            return probs[target_idx]
        else:
            pred = model.predict(X_input)[0]
            return 1.0 if pred == target_idx else 0.0
    except Exception as e:
        print(f"Heuristic evaluation error: {e}")
        return 0.5

def minimax(board, depth, is_maximizing, ai_symbol, opponent_symbol, alpha, beta):
    """Recursive minimax with alpha-beta pruning."""
    res = check_winner(board)
    if res == ai_symbol:
        return 10.0 + depth # Favor quick wins
    if res == opponent_symbol:
        return -10.0 - depth # Delay losses
    if res == 'draw':
        return 0.0
    
    if depth == 0:
        return evaluate_board(board, ai_symbol)
    
    if is_maximizing:
        best_score = -float('inf')
        for i in range(9):
            if board[i].lower() == 'b':
                board[i] = ai_symbol
                score = minimax(board, depth - 1, False, ai_symbol, opponent_symbol, alpha, beta)
                board[i] = 'b'
                best_score = max(score, best_score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
        return best_score
    else:
        best_score = float('inf')
        for i in range(9):
            if board[i].lower() == 'b':
                board[i] = opponent_symbol
                score = minimax(board, depth - 1, True, ai_symbol, opponent_symbol, alpha, beta)
                board[i] = 'b'
                best_score = min(score, best_score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
        return best_score

@app.route('/ai-move', methods=['POST'])
def ai_move():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    board = data.get('board') 
    ai_symbol = data.get('symbol', 'o').lower()
    depth = data.get('depth', 4) # Default to 4 if not provided
    opponent_symbol = 'x' if ai_symbol == 'o' else 'o'
    
    if not board or len(board) != 9:
        return jsonify({'error': 'Invalid board data.'}), 400

    print(f"AI ({ai_symbol}) thinking ahead with depth {depth} using Minimax...")
    
    best_move = -1
    best_score = -float('inf')
    
    # Check all possible moves at the top level
    for i in range(9):
        if board[i].lower() == 'b':
            temp_board = list(board)
            temp_board[i] = ai_symbol
            
            score = minimax(temp_board, depth, False, ai_symbol, opponent_symbol, -float('inf'), float('inf'))
            
            if score > best_score:
                best_score = score
                best_move = i

    if best_move == -1:
        # Fallback to first available if something goes wrong
        empty_indices = [i for i, s in enumerate(board) if s.lower() == 'b']
        best_move = empty_indices[0] if empty_indices else -1

    print(f"AI chose move at index {best_move} (score: {best_score})")
    return jsonify({
        'best_move': int(best_move),
        'score': float(best_score)
    })

@app.route('/save-game', methods=['POST'])
def save_game():
    data = request.json
    board = data.get('board')
    winner = data.get('winner') 
    
    if not board or len(board) != 9 or not winner:
        return jsonify({'error': 'Invalid data.'}), 400

    if winner == 'draw':
        return jsonify({'message': 'Draw games not saved.'}), 200

    result_class = 'positive' if winner == 'x' else 'negative'
    
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="tictactoe"
        )
        cursor = conn.cursor()
        
        # 1. Create the user_game_data table if it doesn't exist
        cols = ['top-left-square', 'top-middle-square', 'top-right-square', 
                'middle-left-square', 'middle-middle-square', 'middle-right-square', 
                'bottom-left-square', 'bottom-middle-square', 'bottom-right-square']
        
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS user_game_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            {', '.join([f'`{c}` VARCHAR(1)' for c in cols])},
            Class VARCHAR(10),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        cursor.execute(create_table_query)

        # 2. Insert the data
        all_cols = cols + ['Class']
        query = f"INSERT INTO user_game_data ({', '.join([f'`{c}`' for c in all_cols])}) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        values = tuple(board + [result_class])
        
        cursor.execute(query, values)
        conn.commit()
        conn.close()
        
        print(f"Game saved to 'user_game_data' as '{result_class}'")
        return jsonify({'status': 'success', 'table': 'user_game_data'})
    except Exception as e:
        print(f"Error saving game: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    return jsonify({'status': 'online', 'model_loaded': model is not None})

@app.route('/retrain', methods=['POST'])
def retrain():
    global model, MODEL_PATH
    try:
        best_name, best_acc, best_path = run_training_pipeline()
        
        # Load the new best model
        MODEL_PATH = best_path
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
        else:
            return jsonify({'error': f'Model saved but file not found at {MODEL_PATH}'}), 500
            
        return jsonify({
            'status': 'success', 
            'best_model': best_name, 
            'accuracy': float(best_acc),
            'message': f'Retrained successfully. New best model is {best_name} with {best_acc:.4f} accuracy.'
        })
    except Exception as e:
        print(f"Error retraining: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
