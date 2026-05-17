import streamlit as st
import numpy as np
import pickle
import os
import time
from main import run_training_pipeline

# --- CONFIG ---
st.set_page_config(page_title="Tic-Tac-Toe", page_icon="🤖", layout="wide")

# Premium Custom CSS for the UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body {
        font-family: 'Outfit', sans-serif;
        max-width: 100vw !important;
        overflow-x: hidden !important;
    }

    /* Main Background & Glassmorphism container approximation */
    .stApp {
        background: radial-gradient(circle at top right, #1e293b 0%, #0b1120 100%);
        max-width: 100vw !important;
        overflow-x: hidden !important;
    }

    /* Title Gradient */
    h1 {
        background: linear-gradient(135deg, #60a5fa 0%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        text-align: center;
        padding-top: 20px;
    }
    
    h3 {
        color: #94a3b8 !important;
        text-align: center;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* Grid Buttons (Square format) targeted specifically */
    div[data-testid="stHorizontalBlock"] div.stButton,
    div[data-testid="stHorizontalBlock"] button,
    button[kind="secondary"] {
        width: 100% !important;
        aspect-ratio: 1 / 1 !important;
        height: auto !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Deep reset inner elements of button to prevent nested spacing issues */
    div[data-testid="stHorizontalBlock"] button * {
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
    }

    button[kind="secondary"] {
        font-size: 3.5rem !important;
        font-weight: 700 !important;
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 16px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: inset 0 0 15px rgba(255, 255, 255, 0.02) !important;
    }
    
    button[kind="secondary"] p {
        font-size: 3.5rem !important;
        margin: 0 !important;
        line-height: 1 !important;
    }
    
    button[kind="secondary"]:hover {
        background: rgba(51, 65, 85, 0.6) !important;
        transform: translateY(-2px) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* Action Buttons (Primary) */
    button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        border: none !important;
        color: white !important;
        font-size: 1.1rem !important;
        border-radius: 12px !important;
        height: 50px !important;
        font-weight: 600 !important;
        box-shadow: 0 8px 20px -5px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.3s !important;
    }
    button[kind="primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 15px 25px -10px rgba(37, 99, 235, 0.5) !important;
    }

    /* Center and restrict the width of the board rows globally */
    div[data-testid="stHorizontalBlock"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: nowrap !important;
        width: 100% !important;
        max-width: 320px !important; /* Perfect desktop size */
        margin: 0 auto !important; /* Center the grid container */
        gap: 8px !important;
    }

    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
        width: 33.33% !important;
        min-width: 0 !important;
        max-width: 33.33% !important;
        flex: 1 1 33.33% !important;
    }

    .winner-text {
        font-size: 2.5rem;
        color: #4ade80;
        text-align: center;
        font-weight: 700;
        margin: 20px 0;
        text-shadow: 0 0 20px rgba(74, 222, 128, 0.4);
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    @media (max-width: 768px) {
        /* Disable top header and bottom footer elements to gain 90px vertical space */
        header[data-testid="stHeader"], footer {
            display: none !important;
        }

        /* Center the main container and remove excessive desktop padding/margins */
        div[data-testid="stAppViewBlockContainer"] {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            max-width: 100% !important;
            min-width: auto !important;
            margin: 0 auto !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
        }

        /* Center the overall vertical block inside the container */
        div[data-testid="stVerticalBlock"] {
            width: 100% !important;
            margin: 0 auto !important;
            display: flex !important;
            flex-direction: column !important;
            align-items: center !important;
        }

        /* Keep the horizontal block (columns) compact, responsive, and perfectly centered */
        div[data-testid="stHorizontalBlock"] {
            max-width: 300px !important;
            width: 100% !important;
            margin: 0 auto !important;
            display: flex !important;
            flex-direction: row !important;
            justify-content: center !important;
            align-items: center !important;
            gap: 6px !important;
        }

        /* Eliminate any side gutters/padding on columns inside the board to keep it clean */
        div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
            padding: 0 !important;
        }

        /* Scale button dimensions and text for a premium feel on mobile */
        button[kind="secondary"] {
            font-size: 2.2rem !important;
            border-radius: 12px !important;
        }
        button[kind="secondary"] p {
            font-size: 2.2rem !important;
        }
        
        /* Scale headers down on mobile to prevent massive text scrolling */
        h1 {
            font-size: 1.6rem !important;
            padding-top: 5px !important;
            margin: 0 0 5px 0 !important;
            text-align: center !important;
            width: 100% !important;
        }
        h3 {
            font-size: 1.0rem !important;
            margin: 0 0 0.8rem 0 !important;
            text-align: center !important;
            width: 100% !important;
        }
        
        /* Center all Markdown texts and elements */
        .stMarkdown, .stHeading, div[data-testid="stMarkdownContainer"] {
            text-align: center !important;
            width: 100% !important;
        }
        
        /* Compact Tip Alert margins */
        [data-testid="stNotification"], .stAlert {
            padding: 0.4rem 0.6rem !important;
            margin: 5px 0 !important;
            width: 100% !important;
            max-width: 300px !important;
        }
        [data-testid="stNotification"] p, .stAlert p {
            font-size: 0.85rem !important;
        }
    }
    </style>
""", unsafe_allow_html=True)

# --- MODEL LOADING ---
MODEL_PATH = 'models/best_model.pkl'
if not os.path.exists(MODEL_PATH):
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
def save_game_to_csv(board, winner):
    if winner == 'draw' or not winner: return
    try:
        row = [f"b'{s.lower()}'" for s in board]
        result_class = "b'positive'" if winner == 'x' else "b'negative'"
        row.append(result_class)
        with open('tictactoe_data.csv', 'a', newline='') as f:
            f.write(",".join(row) + "\n")
        print(f"Saved game to CSV: {result_class}")
    except Exception as e:
        print(f"Failed to save to CSV: {e}")

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
    st.session_state.animation_played = False

def reset_game():
    st.session_state.board = ['b'] * 9
    st.session_state.current_turn = 'x'
    st.session_state.winner = None
    st.session_state.animation_played = False

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
    if st.button("🚀 Run Training Pipeline", type="primary", use_container_width=True):
        with st.spinner("Retraining models..."):
            best_name, best_acc, best_path, results, total_rows = run_training_pipeline()
            st.session_state.retrain_stats = {
                "best_name": best_name,
                "best_acc": best_acc,
                "total_rows": total_rows,
                "results": results
            }
            st.cache_resource.clear() # Force reload model
            st.rerun()

    if os.path.exists("plots/accuracy_comparison.png"):
        st.header("4. Model Analytics")
        st.image("plots/accuracy_comparison.png", caption="Model Accuracy Comparison")
        st.image("plots/confusion_matrices.png", caption="Confusion Matrices")

# --- MAIN UI ---
if "retrain_stats" in st.session_state:
    stats = st.session_state.retrain_stats
    if hasattr(st, "dialog"):
        @st.dialog("🔄 Retraining Complete!")
        def show_stats_dialog(stats_data):
            st.markdown(f"### 📊 Total Dataset Size: **{stats_data['total_rows']} rows**")
            st.divider()
            st.markdown("### 📈 Model Performance Breakdown:")
            for model_name, res in stats_data['results'].items():
                acc_percentage = res['accuracy'] * 100
                st.markdown(f"💡 **{model_name}**: `{acc_percentage:.2f}%` accuracy")
            st.divider()
            st.success(f"🏆 **Winning Model Deployed:** {stats_data['best_name']} ({stats_data['best_acc']*100:.2f}%)")
            if st.button("Got it!", type="primary", use_container_width=True):
                del st.session_state.retrain_stats
                st.rerun()
        show_stats_dialog(stats)
    else:
        st.warning("🔄 Retraining Complete! See stats below:")
        with st.container(border=True):
            st.subheader("📊 MLOps Retraining Statistics")
            st.markdown(f"**Total Dataset Size:** `{stats['total_rows']}` rows")
            for model_name, res in stats['results'].items():
                acc_percentage = res['accuracy'] * 100
                st.markdown(f"- **{model_name}**: `{acc_percentage:.2f}%` accuracy")
            st.success(f"🏆 **Selected Model:** {stats['best_name']} ({stats['best_acc']*100:.2f}%)")
            if st.button("Close Stats Panel", type="primary", use_container_width=True):
                del st.session_state.retrain_stats
                st.rerun()

st.title("🎮 Tic-Tac-Toe by Fawad")
st.subheader("Play against a Machine Learning powered Minimax AI")

if st.session_state.winner:
    if not st.session_state.get('animation_played', False):
        if st.session_state.winner == 'x':
            firecracker_html = """
            <style>
            .firecracker {
                position: fixed;
                font-size: 6rem;
                z-index: 9999;
                pointer-events: none;
                animation: explode 2.5s ease-out forwards;
            }
            @keyframes explode {
                0% { transform: scale(0) translateY(100vh) rotate(0deg); opacity: 1; }
                50% { transform: scale(1.5) translateY(20vh) rotate(180deg); opacity: 1; }
                100% { transform: scale(2) translateY(-10vh) rotate(360deg); opacity: 0; }
            }
            </style>
            <div class="firecracker" style="left: 20%; animation-delay: 0s;">🎆</div>
            <div class="firecracker" style="left: 50%; animation-delay: 0.2s;">🎇</div>
            <div class="firecracker" style="left: 80%; animation-delay: 0.4s;">🎆</div>
            <div class="firecracker" style="left: 35%; animation-delay: 0.6s;">🎇</div>
            <div class="firecracker" style="left: 65%; animation-delay: 0.3s;">🎆</div>
            """
            st.markdown(firecracker_html, unsafe_allow_html=True)
            st.balloons()
        elif st.session_state.winner == 'o':
            st.snow()
        st.session_state.animation_played = True

    if st.session_state.winner == 'draw':
        st.markdown('<p class="winner-text">It\'s a Draw! 🤝</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<p class="winner-text">{"You" if st.session_state.winner == "x" else "AI"} Won! 🎉</p>', unsafe_allow_html=True)
    if st.button("Play Again", type="primary", use_container_width=True):
        reset_game()
        st.rerun()
else:
    st.markdown(f"<h3>Current Turn: <b>{'You (X)' if st.session_state.current_turn == 'x' else 'AI (O)'}</b></h3>", unsafe_allow_html=True)

# Grid Layout
st.write("") # Add some spacing
grid_container = st.container()

with grid_container:
    for row in range(3):
        cols = st.columns(3)
        for col in range(3):
            i = row * 3 + col
            with cols[col]:
                val = st.session_state.board[i]
                if val == 'x':
                    label = "**:blue[X]**"
                elif val == 'o':
                    label = "**:green[O]**"
                else:
                    label = " "
                
                if st.button(label, key=f"btn_{i}", use_container_width=True, disabled=st.session_state.board[i] != 'b' or st.session_state.winner is not None):
                    if st.session_state.current_turn == 'x':
                        st.session_state.board[i] = 'x'
                        st.session_state.winner = check_winner(st.session_state.board)
                        if not st.session_state.winner:
                            st.session_state.current_turn = 'o'
                        else:
                            if st.session_state.winner == 'x': st.session_state.scores['Player'] += 1
                            elif st.session_state.winner == 'draw': st.session_state.scores['Draws'] += 1
                            
                            if st.session_state.winner != 'draw':
                                save_game_to_csv(st.session_state.board, st.session_state.winner)
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
            
            if st.session_state.winner != 'draw':
                save_game_to_csv(st.session_state.board, st.session_state.winner)
        st.rerun()

st.divider()
st.info("💡 **Tip:** You can retrain the AI models using the button in the sidebar to see how accuracy improves with new data!")
