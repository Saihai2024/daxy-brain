CREATE TABLE IF NOT EXISTS sentence_corpus (
    id          SERIAL PRIMARY KEY,
    topic       VARCHAR(255) UNIQUE,
    content     TEXT,
    status      VARCHAR(50) DEFAULT 'pending',
    credibility REAL DEFAULT 1.0,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS vision_corpus (
    id              SERIAL PRIMARY KEY,
    image_path      VARCHAR(255) NOT NULL,
    question        TEXT,
    answer          TEXT,
    surprise_score  REAL,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status          VARCHAR(20) DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS web_corpus (
    id          SERIAL PRIMARY KEY,
    topic       VARCHAR(255) UNIQUE,
    content     TEXT,
    status      VARCHAR(50) DEFAULT 'pending',
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    credibility REAL DEFAULT 1.0
);

CREATE TABLE IF NOT EXISTS training_log (
    id          SERIAL PRIMARY KEY,
    ts          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    task        VARCHAR(32) NOT NULL DEFAULT 'multimodal',
    record_id   INT,
    image_path  VARCHAR(255),
    epoch       INT,
    loss        DOUBLE PRECISION,
    note        VARCHAR(255)
);

CREATE INDEX IF NOT EXISTS idx_training_log_ts   ON training_log (ts);
CREATE INDEX IF NOT EXISTS idx_training_log_task ON training_log (task);
CREATE INDEX IF NOT EXISTS idx_sentence_status   ON sentence_corpus (status);
CREATE INDEX IF NOT EXISTS idx_web_status        ON web_corpus (status);
CREATE INDEX IF NOT EXISTS idx_vision_status     ON vision_corpus (status);
