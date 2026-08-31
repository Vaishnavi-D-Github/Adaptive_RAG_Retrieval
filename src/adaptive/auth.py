"""Authentication services for the web application layer.

This module owns user authentication only. It deliberately does not create or
alter database tables; the project database is expected to provide the existing
``users`` table.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import pymysql
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError, VerificationError
from argon2.low_level import Type


VALID_ROLES = {"employee", "hr"}


class AuthError(ValueError):
    """Raised for validation or authentication failures."""


class DuplicateEmailError(AuthError):
    """Raised when a normalized email is already registered."""


@dataclass(frozen=True)
class SafeUser:
    user_id: str
    full_name: str
    email: str
    role: str
    is_active: bool

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "full_name": self.full_name,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
        }


def normalize_email(email: str) -> str:
    return str(email or "").strip().lower()


def validate_role(role: str) -> str:
    normalized = str(role or "").strip().lower()
    if normalized not in VALID_ROLES:
        raise AuthError("Role must be employee or hr.")
    return normalized


class MySQLAuthService:
    """User service backed by the existing MySQL ``users`` table."""

    def __init__(
        self,
        *,
        host: Optional[str] = None,
        port: Optional[int] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        password_hasher: Optional[PasswordHasher] = None,
    ):
        self.host = host or os.getenv("AE_RAG_DB_HOST", "localhost")
        self.port = int(port or os.getenv("AE_RAG_DB_PORT", "3306"))
        self.user = user or os.getenv("AE_RAG_DB_USER", "root")
        self.password = password if password is not None else os.getenv("AE_RAG_DB_PASSWORD", "")
        self.database = database or os.getenv("AE_RAG_DB_NAME", "adaptive_enterprise_rag")
        self.password_hasher = password_hasher or PasswordHasher(type=Type.ID)

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
    def _safe_user(row: dict) -> SafeUser:
        return SafeUser(
            user_id=str(row["user_id"]),
            full_name=str(row["full_name"]),
            email=str(row["email"]),
            role=str(row["role"]).lower(),
            is_active=bool(row["is_active"]),
        )

    def create_user(self, *, full_name: str, email: str, password: str, role: str) -> SafeUser:
        name = str(full_name or "").strip()
        normalized_email = normalize_email(email)
        normalized_role = validate_role(role)
        if not name:
            raise AuthError("Full name is required.")
        if not normalized_email:
            raise AuthError("Email is required.")
        if not str(password or ""):
            raise AuthError("Password is required.")

        user_id = str(uuid.uuid4())
        password_hash = self.password_hasher.hash(str(password))
        now = datetime.now()
        sql = """
        INSERT INTO users (
            user_id, full_name, email, password_hash, role, is_active, created_at, updated_at
        )
        VALUES (%s, %s, %s, %s, %s, TRUE, %s, %s)
        """
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, (user_id, name, normalized_email, password_hash, normalized_role, now, now))
            connection.commit()
        except pymysql.err.IntegrityError as error:
            connection.rollback()
            raise DuplicateEmailError("An account with this email already exists.") from error
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

        return SafeUser(user_id, name, normalized_email, normalized_role, True)

    def authenticate_user(self, *, email: str, password: str) -> SafeUser:
        normalized_email = normalize_email(email)
        if not normalized_email or not str(password or ""):
            raise AuthError("Invalid credentials.")

        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT user_id, full_name, email, password_hash, role, is_active
                    FROM users
                    WHERE email = %s
                    LIMIT 1
                    """,
                    (normalized_email,),
                )
                row = cursor.fetchone()
                if not row:
                    raise AuthError("Invalid credentials.")
                if not bool(row["is_active"]):
                    raise AuthError("Account disabled.")
                try:
                    verified = self.password_hasher.verify(row["password_hash"], str(password))
                except (VerifyMismatchError, VerificationError):
                    verified = False
                if not verified:
                    raise AuthError("Invalid credentials.")
                if self.password_hasher.check_needs_rehash(row["password_hash"]):
                    cursor.execute(
                        "UPDATE users SET password_hash = %s, updated_at = %s WHERE user_id = %s",
                        (self.password_hasher.hash(str(password)), datetime.now(), row["user_id"]),
                    )
                    connection.commit()
                return self._safe_user(row)
        finally:
            connection.close()

    def get_user_by_id(self, user_id: str) -> Optional[SafeUser]:
        if not str(user_id or "").strip():
            return None
        connection = self._connect()
        try:
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT user_id, full_name, email, role, is_active
                    FROM users
                    WHERE user_id = %s
                    LIMIT 1
                    """,
                    (str(user_id),),
                )
                row = cursor.fetchone()
                return self._safe_user(row) if row else None
        finally:
            connection.close()
