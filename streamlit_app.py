import streamlit as st
import numpy as np
import pickle
import os
import time
from main import run_training_pipeline

# --- CONFIG ---
st.set_page_config(page_title="Tic-Tac-Toe AI by fawad", page_icon="🤖", layout="wide")

# Custom CSS for the grid
st.markdown("""
    <style>
    .stButton>button {
        height: 100px;
        width: 100px;
        font-size: 30px;
        font-weight: bold;
    }
    .winner-text {
        font-size: 40px;
        color: #4ade80;
        text-align: center;
        font-weight: bold;
    }
    .ai-turn {
        color: #fbbf24;
    }
    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
MODEL_PATH = 'models/dt_model.pkl'
FEATURE_MAP = {'b': 0, 'o': 1, 'x': 2}
TARGET_MAP = {0: 'negative', 1: 'positive'}

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

model = load_model(MODEL_PATH)

# --- GAME LOGIC ---
def check_winner(board):
    lines = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]
    for a, b, c in lines:
        if board[a] != 'b' and board[a] == board[b] == board[c]:
            return board[a]
    if 'b' not in board:
        return 'draw'
    return None

def evaluate_board(board, ai_symbol):
    if model is None: return 0.5
    try:
        numeric_board = [FEATURE_MAP[s.lower()] for s in board]
        X_input = np.array([numeric_board])
        target_idx = 1 if ai_symbol == 'x' else 0
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X_input)[0]
            return probs[target_idx]
        else:
            pred = model.predict(X_input)[0]
            return 1.0 if pred == target_idx else 0.0
    except:
        return 0.5

def minimax(board, depth, is_maximizing, ai_symbol, opponent_symbol, alpha, beta):
    res = check_winner(board)
    if res == ai_symbol: return 10.0 + depth
    if res == opponent_symbol: return -10.0 - depth
    if res == 'draw': return 0.0
    
    if depth == 0:
        return evaluate_board(board, ai_symbol)
    
    if is_maximizing:
        best_score = -float('inf')
        for i in range(9):
            if board[i] == 'b':
                board[i] = ai_symbol
                score = minimax(board, depth - 1, False, ai_symbol, opponent_symbol, alpha, beta)
                board[i] = 'b'
                best_score = max(score, best_score)
                alpha = max(alpha, best_score)
                if beta <= alpha: break
        return best_score
    else:
        best_score = float('inf')
        for i in range(9):
            if board[i] == 'b':
                board[i] = opponent_symbol
                score = minimax(board, depth - 1, True, ai_symbol, opponent_symbol, alpha, beta)
                board[i] = 'b'
                best_score = min(score, best_score)
                beta = min(beta, best_score)
                if beta <= alpha: break
        return best_score

def get_ai_move(board, ai_symbol, depth):
    opponent_symbol = 'x' if ai_symbol == 'o' else 'o'
    best_move = -1
    best_score = -float('inf')
    
    for i in range(9):
        if board[i] == 'b':
            temp_board = list(board)
            temp_board[i] = ai_symbol
            score = minimax(temp_board, depth, False, ai_symbol, opponent_symbol, -float('inf'), float('inf'))
            if score > best_score:
                best_score = score
                best_move = i
    return best_move

# --- SESSION STATE ---
if 'board' not in st.session_state:
    st.session_state.board = ['b'] * 9
    st.session_state.current_turn = 'x'
    st.session_state.winner = None
    st.session_state.scores = {'Player': 0, 'AI': 0, 'Draws': 0}

def reset_game():
    st.session_state.board = ['b'] * 9
    st.session_state.current_turn = 'x'
    st.session_state.winner = None

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ MLOps Dashboard")
    
    st.header("1. Game Settings")
    difficulty = st.select_slider("AI Difficulty (Depth)", options=["Easy", "Medium", "Hard"], value="Medium")
    depth_map = {"Easy": 1, "Medium": 3, "Hard": 6}
    
    st.header("2. Statistics")
    st.write(f"🏆 Player: {st.session_state.scores['Player']}")
    st.write(f"🤖 AI: {st.session_state.scores['AI']}")
    st.write(f"🤝 Draws: {st.session_state.scores['Draws']}")
    
    st.header("3. Retraining Pipeline")
    if st.button("🚀 Run Training Pipeline"):
        with st.spinner("Retraining models..."):
            best_name, best_acc, best_path = run_training_pipeline()
            st.success(f"Best Model: {best_name}")
            st.info(f"Accuracy: {best_acc:.4f}")
            st.cache_resource.clear() # Force reload model
            st.rerun()

    if os.path.exists("plots/accuracy_comparison.png"):
        st.header("4. Model Analytics")
        st.image("plots/accuracy_comparison.png", caption="Model Accuracy Comparison")
        st.image("plots/confusion_matrices.png", caption="Confusion Matrices")

# --- MAIN UI ---
st.title("🎮 Tic-Tac-Toe AI")
st.subheader("Play against a Machine Learning powered Minimax AI")

if st.session_state.winner:
    if st.session_state.winner == 'draw':
        st.markdown('<p class="winner-text">It\'s a Draw! 🤝</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="winner-text">{"You" if st.session_state.winner == "x" else "AI"} Won! 🎉</p>', unsafe_allow_html=True)
    if st.button("Play Again"):
        reset_game()
        st.rerun()
else:
    st.write(f"Current Turn: **{'You (X)' if st.session_state.current_turn == 'x' else 'AI (O)'}**")

# Grid Layout
cols = st.columns([1, 1, 1])
for i in range(9):
    with cols[i % 3]:
        label = st.session_state.board[i].upper() if st.session_state.board[i] != 'b' else ""
        if st.button(label if label else " ", key=f"btn_{i}", disabled=st.session_state.board[i] != 'b' or st.session_state.winner is not None):
            if st.session_state.current_turn == 'x':
                st.session_state.board[i] = 'x'
                st.session_state.winner = check_winner(st.session_state.board)
                if not st.session_state.winner:
                    st.session_state.current_turn = 'o'
                else:
                    if st.session_state.winner == 'x': st.session_state.scores['Player'] += 1
                    elif st.session_state.winner == 'draw': st.session_state.scores['Draws'] += 1
                st.rerun()

# AI Move
if st.session_state.current_turn == 'o' and not st.session_state.winner:
    with st.spinner("AI is thinking..."):
        time.sleep(0.5) # Add a small delay for realism
        move = get_ai_move(st.session_state.board, 'o', depth_map[difficulty])
        if move != -1:
            st.session_state.board[move] = 'o'
        st.session_state.winner = check_winner(st.session_state.board)
        if not st.session_state.winner:
            st.session_state.current_turn = 'x'
        else:
            if st.session_state.winner == 'o': st.session_state.scores['AI'] += 1
            elif st.session_state.winner == 'draw': st.session_state.scores['Draws'] += 1
        st.rerun()

st.divider()
st.info("💡 **Tip:** You can retrain the AI models using the button in the sidebar to see how accuracy improves with new data!")
