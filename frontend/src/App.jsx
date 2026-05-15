import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { 
  RefreshCcw, User, Cpu, RotateCcw, Hash, Info, 
  Database, Zap, Trophy, Activity, Brain, Target, ShieldCheck
} from 'lucide-react';
import './App.css';

const API_URL = 'http://localhost:5000';

function App() {
  const [board, setBoard] = useState(Array(9).fill('b'));
  const [playerSymbol, setPlayerSymbol] = useState('x');
  const [aiSymbol, setAiSymbol] = useState('o');
  const [currentTurn, setCurrentTurn] = useState('x');
  const [winner, setWinner] = useState(null);
  const [scores, setScores] = useState({ player: 0, ai: 0, draws: 0 });
  const [loading, setLoading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("Your turn! (X)");
  const [gameCount, setGameCount] = useState(1);
  const [dataSaved, setDataSaved] = useState(false);
  const [aiAnalysis, setAiAnalysis] = useState(null);
  const [lastMove, setLastMove] = useState(null);
  const [difficulty, setDifficulty] = useState('medium');
  const [retrainResult, setRetrainResult] = useState(null);
  const [isRetraining, setIsRetraining] = useState(false);

  const handleRetrain = async () => {
    setIsRetraining(true);
    setRetrainResult("Retraining in progress...");
    try {
      const response = await axios.post(`${API_URL}/retrain`);
      if (response.data.status === 'success') {
        setRetrainResult(`Success! Best: ${response.data.best_model} (${(response.data.accuracy * 100).toFixed(1)}%)`);
      } else {
        setRetrainResult(`Error: ${response.data.error}`);
      }
    } catch (err) {
      console.error("Retrain failed:", err);
      setRetrainResult("Retraining failed.");
    } finally {
      setIsRetraining(false);
    }
  };


  const difficultyMap = {
    easy: 1,
    medium: 3,
    hard: 6
  };

  const checkWinner = (squares) => {
    const lines = [
      [0, 1, 2], [3, 4, 5], [6, 7, 8],
      [0, 3, 6], [1, 4, 7], [2, 5, 8],
      [0, 4, 8], [2, 4, 6]
    ];
    for (let line of lines) {
      const [a, b, c] = line;
      if (squares[a] !== 'b' && squares[a] === squares[b] && squares[a] === squares[c]) {
        return squares[a];
      }
    }
    if (!squares.includes('b')) return 'draw';
    return null;
  };

  const handlePlayerMove = (idx) => {
    if (board[idx] !== 'b' || currentTurn !== playerSymbol || winner) return;

    const newBoard = [...board];
    newBoard[idx] = playerSymbol;
    setBoard(newBoard);
    setLastMove(idx);
    
    const result = checkWinner(newBoard);
    if (result) {
      endGame(result, newBoard);
    } else {
      setCurrentTurn(aiSymbol);
      setStatusMessage(`AI is thinking...`);
    }
  };

  useEffect(() => {
    if (currentTurn === aiSymbol && !winner) {
      const timeout = setTimeout(makeAiMove, 600);
      return () => clearTimeout(timeout);
    }
  }, [currentTurn, winner]);

  const makeAiMove = async () => {
    setLoading(true);
    try {
      const response = await axios.post(`${API_URL}/ai-move`, { 
        board, 
        symbol: aiSymbol,
        depth: difficultyMap[difficulty]
      });
      const { best_move, score } = response.data;
      setAiAnalysis(score);
      setLastMove(best_move);
      
      const newBoard = [...board];
      newBoard[best_move] = aiSymbol;
      setBoard(newBoard);

      const result = checkWinner(newBoard);
      if (result) {
        endGame(result, newBoard);
      } else {
        setCurrentTurn(playerSymbol);
        setStatusMessage(`Your turn! (${playerSymbol.toUpperCase()})`);
      }
    } catch (err) {
      console.error("AI Move failed:", err);
      setStatusMessage(`Error: Connection lost`);
    } finally {
      setLoading(false);
    }
  };

  const endGame = async (result, finalBoard) => {
    setWinner(result);
    setDataSaved(false);
    
    if (result === playerSymbol) {
      setScores(s => ({ ...s, player: s.player + 1 }));
      setStatusMessage("You won! 🎉");
    } else if (result === aiSymbol) {
      setScores(s => ({ ...s, ai: s.ai + 1 }));
      setStatusMessage("AI won! 🤖");
    } else {
      setScores(s => ({ ...s, draws: s.draws + 1 }));
      setStatusMessage("It's a draw! 🤝");
    }

    try {
      await axios.post(`${API_URL}/save-game`, { 
        board: finalBoard, 
        winner: result 
      });
      setDataSaved(true);
    } catch (err) {
      console.error("Failed to save game data:", err);
    }
  };

  const resetGame = () => {
    const nextPlayerSymbol = playerSymbol === 'x' ? 'o' : 'x';
    const nextAiSymbol = aiSymbol === 'x' ? 'o' : 'x';
    
    setPlayerSymbol(nextPlayerSymbol);
    setAiSymbol(nextAiSymbol);
    setBoard(Array(9).fill('b'));
    setWinner(null);
    setGameCount(prev => prev + 1);
    setDataSaved(false);
    setAiAnalysis(null);
    setLastMove(null);

    setCurrentTurn('x');
    setStatusMessage(nextPlayerSymbol === 'x' ? "Your turn! (X)" : "AI starts (X)...");
  };

  return (
    <div className="container">
      <div className="game-layout">
        
        {/* Sidebar Left */}
        <aside className="sidebar">
          <div className="sidebar-card">
            <h2 className="sidebar-section-title">
              <ShieldCheck className="tag-icon" /> AI Difficulty
            </h2>
            <div className="difficulty-selector">
              {Object.keys(difficultyMap).map((level) => (
                <button
                  key={level}
                  onClick={() => setDifficulty(level)}
                  className={`diff-btn ${difficulty === level ? 'active' : ''}`}
                >
                  {level}
                </button>
              ))}
            </div>
          </div>

          <div className="sidebar-card">
            <h2 className="sidebar-section-title">
              <Trophy className="tag-icon" /> Performance
            </h2>
            <div className="score-board">
              <div className="score-item">
                <div className="score-info">
                  <User className="score-icon player" />
                  <span className="score-label">Player ({playerSymbol.toUpperCase()})</span>
                </div>
                <span className="score-value">{scores.player}</span>
              </div>
              <div className="score-item">
                <div className="score-info">
                  <Cpu className="score-icon ai" />
                  <span className="score-label">AI ({aiSymbol.toUpperCase()})</span>
                </div>
                <span className="score-value">{scores.ai}</span>
              </div>
              <div className="score-item">
                <div className="score-info">
                  <Hash className="score-icon draw" />
                  <span className="score-label">Draws</span>
                </div>
                <span className="score-value">{scores.draws}</span>
              </div>
            </div>
          </div>

          <div className="sidebar-card">
            <h2 className="sidebar-section-title">
              <Brain className="tag-icon" /> AI Analysis
            </h2>
            <div className="footer-info">
              <div className="tag-item">
                <Activity className="tag-icon" />
                <span>Strategy: Minimax (Depth {difficultyMap[difficulty]})</span>
              </div>
              <div className="tag-item">
                <Target className="tag-icon" />
                <span>Confidence: {aiAnalysis !== null ? `${aiAnalysis.toFixed(2)}` : 'Waiting...'}</span>
              </div>
            </div>
          </div>

          <div className="sidebar-card">
            <h2 className="sidebar-section-title">
              <Database className="tag-icon" /> Data Pipeline
            </h2>
            {dataSaved ? (
              <div className="data-badge">
                <Zap className="tag-icon" />
                <span>State Synchronized</span>
              </div>
            ) : (
              <div className="tag-item" style={{padding: '10px 0'}}>
                <Info className="tag-icon" />
                <span style={{fontSize: '0.75rem'}}>Waiting for game end...</span>
              </div>
            )}
            
            <button 
              className="action-btn" 
              style={{marginTop: '15px', width: '100%', fontSize: '0.9rem', padding: '10px'}}
              onClick={handleRetrain}
              disabled={isRetraining}
            >
              <RefreshCcw className={`icon ${isRetraining ? 'spin' : ''}`} />
              <span>{isRetraining ? 'Retraining...' : 'Retrain Models'}</span>
            </button>
            {retrainResult && (
               <div style={{marginTop: '10px', fontSize: '0.8rem', color: '#10b981', textAlign: 'center'}}>
                 {retrainResult}
               </div>
            )}
          </div>
        </aside>

        {/* Main Content Right */}
        <main className="main-content">
          <header>
            <div className="title-wrapper" style={{display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px', marginBottom: '8px'}}>
              <Brain className="title-icon" style={{width: '32px', height: '32px', color: '#60a5fa'}} />
              <h1 className="title" style={{marginBottom: 0}}>Tic Tac Toe AI</h1>
            </div>
            <p className="subtitle">MLOps Powered Gameplay • Round {gameCount}</p>
          </header>

          <div className={`status-banner ${winner ? 'game-over' : ''}`}>
            <div className={`status-dot ${currentTurn === playerSymbol ? 'active' : (loading ? 'thinking' : '')}`}></div>
            {loading && <Brain className="icon-sm spin" style={{width: '16px', height: '16px', color: '#fbbf24'}} />}
            <span className="status-text">{statusMessage}</span>
          </div>

          <div className="board-wrapper">
            <div className="board">
              {board.map((cell, idx) => (
                <button
                  key={idx}
                  onClick={() => handlePlayerMove(idx)}
                  className={`square ${cell} ${winner ? 'disabled' : ''} ${winner && cell === 'b' ? 'dimmed' : ''} ${lastMove === idx ? 'last-move' : ''}`}
                  disabled={!!winner || currentTurn !== playerSymbol || cell !== 'b'}
                >
                  <span className="symbol-layer">{cell === 'b' ? '' : cell.toUpperCase()}</span>
                </button>
              ))}
            </div>
          </div>

          <button onClick={resetGame} className="action-btn">
            <RotateCcw className={`icon ${winner ? 'pulse' : ''}`} />
            <span>{winner ? 'Next Round' : 'Restart Round'}</span>
          </button>
        </main>

      </div>
    </div>
  );
}

export default App;
