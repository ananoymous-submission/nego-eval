-- Users table: Track all participants
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User preferences table: Store survey responses about negotiation preferences
CREATE TABLE IF NOT EXISTS user_preferences (
    preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    strategy_preference TEXT NOT NULL,
    risk_tolerance INTEGER NOT NULL CHECK (risk_tolerance BETWEEN 0 AND 10),
    outcome_preference TEXT NOT NULL,
    communication_style TEXT NOT NULL,
    problem_approach TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Negotiation sessions table: Track all negotiation sessions
CREATE TABLE IF NOT EXISTS negotiation_sessions (
    session_id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL,
    llm_model_name TEXT NOT NULL,
    human_profile_path TEXT NOT NULL,
    llm_profile_path TEXT NOT NULL,
    status TEXT NOT NULL,  -- 'in_progress', 'completed', 'failed'
    log_path TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON negotiation_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON negotiation_sessions(status);
