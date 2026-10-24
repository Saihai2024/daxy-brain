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

CREATE TABLE IF NOT EXISTS topic_failed_blacklist (
    topic       VARCHAR(255) PRIMARY KEY,
    fail_count  INT DEFAULT 1,
    last_failed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS thinking_corpus (
    id            SERIAL PRIMARY KEY,
    src           TEXT NOT NULL,
    src_id        INTEGER NOT NULL,
    task_text     TEXT NOT NULL,
    task_type     TEXT,
    steps         JSONB NOT NULL DEFAULT '[]'::jsonb,
    tool_plan     JSONB NOT NULL DEFAULT '[]'::jsonb,
    pitfalls      JSONB NOT NULL DEFAULT '[]'::jsonb,
    acceptance    JSONB NOT NULL DEFAULT '[]'::jsonb,
    qwen_thinking JSONB NOT NULL DEFAULT '[]'::jsonb,
    answer_text   TEXT,
    elapsed       REAL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (src, src_id)
);
CREATE INDEX IF NOT EXISTS thinking_corpus_src_idx ON thinking_corpus (src);

CREATE TABLE IF NOT EXISTS thinking_gaps (
    id         SERIAL PRIMARY KEY,
    query      TEXT UNIQUE NOT NULL,
    hits       INTEGER NOT NULL DEFAULT 1,
    best_rel   REAL,
    status     TEXT NOT NULL DEFAULT 'pending',
    fails      INTEGER NOT NULL DEFAULT 0,
    first_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS thinking_gaps_pick_idx
    ON thinking_gaps (status, hits DESC, last_seen DESC);

CREATE TABLE IF NOT EXISTS thinking_seeds (
    id         SERIAL PRIMARY KEY,
    topic      TEXT,
    task_text  TEXT UNIQUE NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending',
    fails      INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS thinking_seeds_pick_idx ON thinking_seeds (status, id);
