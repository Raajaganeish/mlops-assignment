import sqlite3
from datetime import datetime, timedelta

DB_FILE = "logs.db"


def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                input_json TEXT,
                prediction_json TEXT,
                status_code INTEGER,
                error_message TEXT,
                model_type TEXT,
                model_version TEXT
            )
        """
        )
        conn.commit()


def log_prediction(
    input_json: str,
    prediction_json: str,
    status_code: int,
    error_message: str,
    model_type: str,
    model_version: str,
):
    timestamp = datetime.utcnow().isoformat()
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (
                timestamp, input_json, prediction_json,
                status_code, error_message, model_type, model_version
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                timestamp,
                input_json,
                prediction_json,
                status_code,
                error_message,
                model_type,
                model_version,
            ),
        )
        conn.commit()


def get_prediction_stats(start: str = None, end: str = None):
    if not start:
        start = (datetime.utcnow() - timedelta(days=15)).isoformat()
    if not end:
        end = datetime.utcnow().isoformat()

    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT COUNT(*),
                   SUM(CASE WHEN status_code = 200 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN status_code = 400 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN status_code = 422 THEN 1 ELSE 0 END),
                   SUM(CASE WHEN status_code = 500 THEN 1 ELSE 0 END)
            FROM predictions
            WHERE timestamp BETWEEN ? AND ?
        """,
            (start, end),
        )
        total, ok_200, err_400, err_422, err_500 = cursor.fetchone()

        cursor.execute(
            """
            SELECT model_version, COUNT(*)
            FROM predictions
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY model_version
        """,
            (start, end),
        )
        versions = cursor.fetchall()

        cursor.execute(
            """
            SELECT AVG(CAST(json_extract(prediction_json, '$.predicted_price') AS REAL))
            FROM predictions
            WHERE status_code = 200 AND timestamp BETWEEN ? AND ?
        """,
            (start, end),
        )
        avg_price = cursor.fetchone()[0]

    return {
        "start": start,
        "end": end,
        "total_requests": total or 0,
        "success_200": ok_200 or 0,
        "bad_request_400": err_400 or 0,
        "validation_errors_422": err_422 or 0,
        "internal_errors_500": err_500 or 0,
        "avg_predicted_price": round(avg_price, 2) if avg_price else None,
        "model_version_usage": {v: c for v, c in versions},
    }
