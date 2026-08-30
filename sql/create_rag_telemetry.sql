CREATE DATABASE IF NOT EXISTS adaptive_enterprise_rag
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE adaptive_enterprise_rag;

CREATE TABLE IF NOT EXISTS rag_runs (
    run_id CHAR(36) NOT NULL,
    question_id VARCHAR(100) NULL,
    question TEXT NOT NULL,

    run_type VARCHAR(30) NOT NULL DEFAULT 'production',
    mode VARCHAR(30) NOT NULL,
    model VARCHAR(100) NULL,
    config_version VARCHAR(100) NULL,
    timestamp DATETIME(6) NOT NULL,

    initial_k INT NULL,
    predicted_k INT NULL,
    selected_k INT NULL,
    maximum_k INT NULL,
    prediction_confidence DOUBLE NULL,

    retrieval_iterations INT NULL,
    retrieval_strategy VARCHAR(100) NULL,
    k_escalated BOOLEAN NOT NULL DEFAULT FALSE,

    verification_performed BOOLEAN NOT NULL DEFAULT FALSE,
    verification_result BOOLEAN NULL,
    verification_score DOUBLE NULL,
    verification_reason TEXT NULL,

    num_retrieved_chunks INT NULL,
    num_chunks_used INT NULL,
    unique_documents INT NULL,

    prompt_tokens INT NULL,
    generated_tokens INT NULL,
    total_tokens INT NULL,

    retrieval_latency_ms DOUBLE NULL,
    verification_latency_ms DOUBLE NULL,
    optimization_latency_ms DOUBLE NULL,
    generation_latency_ms DOUBLE NULL,
    total_latency_ms DOUBLE NULL,

    ollama_prompt_eval_duration_ms DOUBLE NULL,
    ollama_eval_duration_ms DOUBLE NULL,
    ollama_total_duration_ms DOUBLE NULL,

    generated_answer LONGTEXT NULL,

    query_features JSON NULL,
    predicted_k_probabilities JSON NULL,
    retrieved_sources JSON NULL,

    context_characters INT NULL,
    prompt_characters INT NULL,

    fallback_used BOOLEAN NOT NULL DEFAULT FALSE,

    faithfulness DOUBLE NULL,
    answer_relevancy DOUBLE NULL,
    context_relevance DOUBLE NULL,
    evaluation_status VARCHAR(30) NOT NULL DEFAULT 'not_evaluated',

    created_at DATETIME(6) NOT NULL DEFAULT CURRENT_TIMESTAMP(6),

    PRIMARY KEY (run_id),

    INDEX idx_question_id (question_id),
    INDEX idx_timestamp (timestamp),
    INDEX idx_mode (mode),
    INDEX idx_selected_k (selected_k),
    INDEX idx_run_type (run_type),
    INDEX idx_evaluation_status (evaluation_status)
) ENGINE=InnoDB;