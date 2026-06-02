CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS roles (
    id SERIAL PRIMARY KEY,
    role_name VARCHAR(100) UNIQUE NOT NULL,
    can_upload BOOLEAN DEFAULT FALSE,
    can_query BOOLEAN DEFAULT TRUE,
    can_admin BOOLEAN DEFAULT FALSE
);

INSERT INTO roles (role_name, can_upload, can_query, can_admin)
VALUES
    ('admin', TRUE, TRUE, TRUE),
    ('user', TRUE, TRUE, FALSE),
    ('viewer', FALSE, TRUE, FALSE)
ON CONFLICT (role_name) DO NOTHING;

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    role_id INTEGER REFERENCES roles(id),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    hash TEXT UNIQUE NOT NULL,
    type TEXT NOT NULL,
    ingested_at TIMESTAMP DEFAULT NOW(),
    user_id VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    page INTEGER,
    text_preview TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ingestion_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    user_id VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS memory_store (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),
    importance_score FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS agent_audit (
    id BIGSERIAL PRIMARY KEY,
    task_id VARCHAR(255),
    user_id VARCHAR(255),
    agent TEXT,
    event_type TEXT,
    payload JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS feedback (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    task_id VARCHAR(255),
    rating INTEGER,
    comment TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS audit_logs (
    id BIGSERIAL PRIMARY KEY,
    user_id VARCHAR(255),
    endpoint TEXT,
    response_code INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);
