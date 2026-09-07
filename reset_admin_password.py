from pathlib import Path
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from adaptive.config import load_project_env
from argon2 import PasswordHasher
from argon2.low_level import Type
import pymysql


# Load the project's .env file
load_project_env()

EMAIL = "hr@example.com"
NEW_PASSWORD = "Admin@12345"


db = {
    "host": os.getenv("AE_RAG_DB_HOST", "localhost"),
    "port": int(os.getenv("AE_RAG_DB_PORT", "3306")),
    "user": os.getenv("AE_RAG_DB_USER", "root"),
    "password": os.getenv("AE_RAG_DB_PASSWORD", ""),
    "database": os.getenv("AE_RAG_DB_NAME", "adaptive_enterprise_rag"),
    "charset": "utf8mb4",
}


print("Connecting to MySQL...")
print("Host:", db["host"])
print("Port:", db["port"])
print("User:", db["user"])
print("Database:", db["database"])
print()


hasher = PasswordHasher(type=Type.ID)
password_hash = hasher.hash(NEW_PASSWORD)

connection = pymysql.connect(**db)

try:
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE users
            SET role = 'admin',
                password_hash = %s,
                is_active = TRUE,
                updated_at = NOW()
            WHERE email = %s
            """,
            (password_hash, EMAIL),
        )

        if cursor.rowcount != 1:
            raise RuntimeError(
                f"Expected to update exactly 1 account, "
                f"but updated {cursor.rowcount}."
            )

    connection.commit()

    print("Admin account updated successfully.")
    print("Email:", EMAIL)
    print("Role: admin")
    print("Password:", NEW_PASSWORD)

finally:
    connection.close()