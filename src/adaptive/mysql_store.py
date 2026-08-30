"""MySQL persistence for production RAG execution telemetry."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import pymysql


class MySQLTelemetryStore:
    """Persist RAG execution telemetry in MySQL.

    Database persistence is intentionally kept separate from the RAG
    execution path. Callers should catch persistence errors so that a
    database outage never prevents an answer from being returned.
    """

    def __init__(
        self,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
    ):
        self.host = host or os.getenv(
            "AE_RAG_DB_HOST",
            "localhost",
        )

        self.port = int(
            port
            or os.getenv(
                "AE_RAG_DB_PORT",
                "3306",
            )
        )

        self.user = user or os.getenv(
            "AE_RAG_DB_USER",
            "root",
        )

        self.password = (
            password
            if password is not None
            else os.getenv(
                "AE_RAG_DB_PASSWORD",
                "",
            )
        )

        self.database = database or os.getenv(
            "AE_RAG_DB_NAME",
            "adaptive_enterprise_rag",
        )

    def _connect(self):
        return pymysql.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            autocommit=False,
        )

    @staticmethod
    def _json(value: Any) -> Optional[str]:
        if value is None:
            return None

        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    def save_run(
        self,
        telemetry: Dict[str, Any],
        *,
        question_id: Optional[str] = None,
        mode: str = "adaptive",
        model: Optional[str] = None,
        run_type: str = "production",
        config_version: Optional[str] = None,
        evaluation_status: str = "not_evaluated",
        faithfulness: Optional[float] = None,
        answer_relevancy: Optional[float] = None,
        context_relevance: Optional[float] = None,
    ) -> str:
        """Persist one completed RAG execution and return its run ID."""

        run_id = str(uuid.uuid4())

        timestamp = datetime.now()

        sql = """
        INSERT INTO rag_runs (
            run_id,
            question_id,
            question,
            run_type,
            mode,
            model,
            config_version,
            timestamp,

            initial_k,
            predicted_k,
            selected_k,
            maximum_k,
            prediction_confidence,

            retrieval_iterations,
            retrieval_strategy,
            k_escalated,

            verification_performed,
            verification_result,
            verification_score,
            verification_reason,

            num_retrieved_chunks,
            num_chunks_used,
            unique_documents,

            prompt_tokens,
            generated_tokens,
            total_tokens,

            retrieval_latency_ms,
            verification_latency_ms,
            optimization_latency_ms,
            generation_latency_ms,
            total_latency_ms,

            ollama_prompt_eval_duration_ms,
            ollama_eval_duration_ms,
            ollama_total_duration_ms,

            generated_answer,

            query_features,
            predicted_k_probabilities,
            retrieved_sources,

            context_characters,
            prompt_characters,

            fallback_used,

            faithfulness,
            answer_relevancy,
            context_relevance,
            evaluation_status
        )
        VALUES (
            %s,%s,%s,%s,%s,%s,%s,%s,
            %s,%s,%s,%s,%s,
            %s,%s,%s,
            %s,%s,%s,%s,
            %s,%s,%s,
            %s,%s,%s,
            %s,%s,%s,%s,%s,
            %s,%s,%s,
            %s,
            %s,%s,%s,
            %s,%s,
            %s,
            %s,%s,%s,%s
        )
        """

        values = (
            run_id,
            question_id,
            telemetry.get("query", ""),
            run_type,
            mode,
            model,
            config_version,
            timestamp,

            telemetry.get("initial_k"),
            telemetry.get("predicted_k"),
            telemetry.get("selected_k"),
            telemetry.get("maximum_k"),
            telemetry.get("predicted_k_confidence"),

            telemetry.get("retrieval_iterations"),
            telemetry.get("retrieval_strategy"),
            telemetry.get("k_escalated", False),

            telemetry.get("verification_performed", False),
            telemetry.get("verification_result"),
            telemetry.get("verification_score"),
            telemetry.get("verification_reason"),

            telemetry.get("num_retrieved_chunks"),
            telemetry.get("num_chunks_used"),
            telemetry.get("unique_documents"),

            telemetry.get("prompt_tokens"),
            telemetry.get("generated_tokens"),
            telemetry.get("total_tokens"),

            telemetry.get("retrieval_time_ms"),
            telemetry.get("verification_time_ms"),
            telemetry.get("optimization_time_ms"),
            telemetry.get("generation_time_ms"),
            telemetry.get("total_latency_ms"),

            telemetry.get(
                "ollama_prompt_eval_duration_ms"
            ),
            telemetry.get(
                "ollama_eval_duration_ms"
            ),
            telemetry.get(
                "ollama_total_duration_ms"
            ),

            telemetry.get("generated_answer", ""),

            self._json(
                telemetry.get("query_features")
            ),
            self._json(
                telemetry.get(
                    "predicted_k_probabilities"
                )
            ),
            self._json(
                telemetry.get("retrieved_sources")
            ),

            telemetry.get("context_characters"),
            telemetry.get("prompt_characters"),

            telemetry.get("fallback_used", False),

            faithfulness,
            answer_relevancy,
            context_relevance,
            evaluation_status,
        )

        connection = self._connect()

        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    sql,
                    values,
                )

            connection.commit()

        except Exception:
            connection.rollback()
            raise

        finally:
            connection.close()

        return run_id