import mysql.connector
import pickle
import numpy as np
import random
import os

# Configuration
MODEL_PATH = 'models/dt_model.pkl'
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "tictactoe"
}

# Feature map for the model
FEATURE_MAP = {'b': 0, 'o': 1, 'x': 2}

def load_model():
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            return pickle.load(f)
    return None

def check_winner(squares):
    lines = [[0,1,2],[3,4,5],[6,7,8],[0,3,6],[1,4,7],[2,5,8],[0,4,8],[2,4,6]]
    for a, b, c in lines:
        if squares[a] != 'b' and squares[a] == squares[b] and squares[a] == squares[c]:
            return squares[a]
    if 'b' not in squares: return 'draw'
    return None

def save_to_db(board, winner):
    if winner == 'draw': return
    result_class = 'positive' if winner == 'x' else 'negative'
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cols = ['top-left-square', 'top-middle-square', 'top-right-square', 'middle-left-square', 'middle-middle-square', 'middle-right-square', 'bottom-left-square', 'bottom-middle-square', 'bottom-right-square']
        create_table_query = f"CREATE TABLE IF NOT EXISTS user_game_data (id INT AUTO_INCREMENT PRIMARY KEY, {', '.join([f'`{c}` VARCHAR(1)' for c in cols])}, Class VARCHAR(10), created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        cursor.execute(create_table_query)
        all_cols = cols + ['Class']
        query = f"INSERT INTO user_game_data ({', '.join([f'`{c}`' for c in all_cols])}) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        cursor.execute(query, tuple(board + [result_class]))
        conn.commit()
        conn.close()
    except Exception as e: print(f"DB Error: {e}")

def get_ai_move(model, board, ai_symbol):
    empty_indices = [i for i, s in enumerate(board) if s == 'b']
    best_move = -1
    highest_prob = -1.0
    target_idx = 0 if ai_symbol == 'o' else 1
    
    for idx in empty_indices:
        simulated_board = list(board)
        simulated_board[idx] = ai_symbol
        numeric_board = [FEATURE_MAP[s] for s in simulated_board]
        probs = model.predict_proba([numeric_board])[0]
        if probs[target_idx] > highest_prob:
            highest_prob = probs[target_idx]
            best_move = idx
    return best_move if best_move != -1 else empty_indices[0]

def simulate_100_games():
    model = load_model()
    if not model: return
    
    print("Simulating 100 games at high speed...")
    stats = {'x': 0, 'o': 0, 'draw': 0}
    
    for i in range(100):
        board = ['b'] * 9
        symbols = ['x', 'o']
        random.shuffle(symbols)
        p1, p2 = symbols
        current = 'x'
        
        while True:
            win = check_winner(board)
            if win: break
            
            # AI logic for BOTH players to generate high-quality game data
            # But one uses the model, one can be a mix of random/logic
            if current == p1:
                # Player 1: Randomish for variety
                move = random.choice([i for i, s in enumerate(board) if s == 'b'])
            else:
                # Player 2: Use Model
                move = get_ai_move(model, board, current)
                
            board[move] = current
            current = 'o' if current == 'x' else 'x'
            
        if win != 'draw': stats[win] += 1
        else: stats['draw'] += 1
        save_to_db(board, win)
        
    print(f"Simulation Done. Stats: {stats}")

if __name__ == "__main__":
    simulate_100_games()
