from __future__ import annotations

import re
import sqlite3
import threading
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

LearningItemKind = Literal["word", "collocation", "grammar"]
_VALID_KINDS = {"word", "collocation", "grammar"}
_KANJI = re.compile(r"[\u3400-\u9fff]")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _normalize_term(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


def _normalize_meaning(value: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip().casefold()


class StudyLibrary:
    """SQLite-backed personal library for vocabulary, collocations, and grammar."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self._write_lock = threading.Lock()
        self._initialize()

    def state_for(
        self,
        *,
        kind: LearningItemKind,
        lemma: str,
        reading: str,
        meaning: str,
        context_key: str,
    ) -> dict[str, Any]:
        clean_kind = self._validate_kind(kind)
        key = self._entry_key(clean_kind, lemma, reading)
        with self._connect() as connection:
            return self._state_with_connection(
                connection,
                key=key,
                meaning=meaning,
                context_key=context_key,
            )

    def save(
        self,
        *,
        kind: LearningItemKind,
        lemma: str,
        surface: str,
        reading: str,
        meaning: str,
        context_key: str,
        source_url: str,
        job_id: str,
        sentence_id: str,
        sentence: str,
    ) -> dict[str, Any]:
        clean_kind = self._validate_kind(kind)
        clean_lemma = lemma.strip()
        clean_surface = surface.strip()
        clean_reading = reading.strip()
        clean_meaning = meaning.strip()
        if not clean_lemma or not clean_surface or not clean_meaning or not context_key.strip():
            raise ValueError("学习词库条目缺少原型、文中形式、释义或语境标识")
        key = self._entry_key(clean_kind, clean_lemma, clean_reading)
        display = (
            clean_surface
            if _KANJI.search(clean_surface) and not _KANJI.search(clean_lemma)
            else clean_lemma
        )
        now = _now()
        with self._write_lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT id FROM learning_entries WHERE normalized_key = ?", (key,)
            ).fetchone()
            added_entry = row is None
            if row is None:
                cursor = connection.execute(
                    """
                    INSERT INTO learning_entries(
                        kind, lemma, reading, display, normalized_key, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        clean_kind,
                        clean_lemma,
                        clean_reading,
                        display,
                        key,
                        now,
                        now,
                    ),
                )
                entry_id = int(cursor.lastrowid)
            else:
                entry_id = int(row["id"])
                connection.execute(
                    """
                    UPDATE learning_entries
                    SET display = ?, updated_at = ?
                    WHERE id = ? AND display NOT GLOB '*[㐀-鿿]*'
                          AND ? GLOB '*[㐀-鿿]*'
                    """,
                    (display, now, entry_id, display),
                )

            form_cursor = connection.execute(
                """
                INSERT INTO learning_forms(entry_id, surface, reading, normalized_surface)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(entry_id, normalized_surface) DO NOTHING
                """,
                (entry_id, clean_surface, clean_reading, _normalize_term(clean_surface)),
            )
            normalized_meaning = _normalize_meaning(clean_meaning)
            meaning_row = connection.execute(
                """
                SELECT id FROM learning_meanings
                WHERE entry_id = ? AND normalized_meaning = ?
                """,
                (entry_id, normalized_meaning),
            ).fetchone()
            added_meaning = meaning_row is None
            if meaning_row is None:
                meaning_cursor = connection.execute(
                    """
                    INSERT INTO learning_meanings(
                        entry_id, meaning, normalized_meaning, first_context, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (entry_id, clean_meaning, normalized_meaning, sentence, now),
                )
                meaning_id = int(meaning_cursor.lastrowid)
            else:
                meaning_id = int(meaning_row["id"])

            encounter_cursor = connection.execute(
                """
                INSERT INTO learning_encounters(
                    entry_id, meaning_id, context_key, source_url, job_id,
                    sentence_id, sentence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(entry_id, context_key) DO NOTHING
                """,
                (
                    entry_id,
                    meaning_id,
                    context_key,
                    source_url,
                    job_id,
                    sentence_id,
                    sentence,
                    now,
                ),
            )
            added_encounter = encounter_cursor.rowcount > 0
            if added_entry or added_meaning or added_encounter or form_cursor.rowcount > 0:
                connection.execute(
                    "UPDATE learning_entries SET updated_at = ? WHERE id = ?",
                    (now, entry_id),
                )
            state = self._state_with_connection(
                connection,
                key=key,
                meaning=clean_meaning,
                context_key=context_key,
            )
            connection.commit()
        return state | {
            "added_entry": added_entry,
            "added_meaning": added_meaning,
            "added_encounter": added_encounter,
        }

    def context_for_analysis(self, sentence: str, limit: int = 80) -> list[dict[str, Any]]:
        """Return compact existing-library context for the language model."""

        clean_sentence = _normalize_term(sentence)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT e.id, e.kind, e.lemma, e.reading, e.display, e.updated_at,
                       COUNT(DISTINCT x.id) AS encounter_count
                FROM learning_entries AS e
                LEFT JOIN learning_encounters AS x ON x.entry_id = e.id
                GROUP BY e.id
                ORDER BY encounter_count DESC, e.updated_at DESC
                LIMIT 240
                """
            ).fetchall()
            if not rows:
                return []
            entry_ids = [int(row["id"]) for row in rows]
            placeholders = ",".join("?" for _ in entry_ids)
            meanings: dict[int, list[str]] = {entry_id: [] for entry_id in entry_ids}
            for row in connection.execute(
                f"""
                SELECT entry_id, meaning FROM learning_meanings
                WHERE entry_id IN ({placeholders}) ORDER BY id
                """,  # noqa: S608 - placeholders are generated, not user input
                entry_ids,
            ):
                meanings[int(row["entry_id"])].append(str(row["meaning"]))
            forms: dict[int, list[str]] = {entry_id: [] for entry_id in entry_ids}
            for row in connection.execute(
                f"""
                SELECT entry_id, surface FROM learning_forms
                WHERE entry_id IN ({placeholders}) ORDER BY id
                """,  # noqa: S608 - placeholders are generated, not user input
                entry_ids,
            ):
                forms[int(row["entry_id"])].append(str(row["surface"]))

        ranked: list[tuple[bool, int, str, dict[str, Any]]] = []
        for row in rows:
            entry_id = int(row["id"])
            item_forms = [str(row["lemma"]), *forms[entry_id]]
            exact_match = any(
                normalized and normalized in clean_sentence
                for normalized in (_normalize_term(value).replace("～", "") for value in item_forms)
            )
            item = {
                "kind": str(row["kind"]),
                "lemma": str(row["lemma"]),
                "reading": str(row["reading"]),
                "meanings": meanings[entry_id],
                "forms": forms[entry_id],
            }
            ranked.append(
                (
                    exact_match,
                    int(row["encounter_count"]),
                    str(row["updated_at"]),
                    item,
                )
            )
        ranked.sort(key=lambda value: (value[0], value[1], value[2]), reverse=True)
        return [item for _, _, _, item in ranked[: max(1, limit)]]

    def list_entries(self) -> list[dict[str, Any]]:
        """Return library entries with meanings and unique sentence encounter counts."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT e.id, e.kind, e.lemma, e.reading, e.display, e.updated_at,
                       COUNT(DISTINCT x.id) AS encounter_count
                FROM learning_entries AS e
                LEFT JOIN learning_encounters AS x ON x.entry_id = e.id
                GROUP BY e.id
                ORDER BY e.updated_at DESC, e.id DESC
                """
            ).fetchall()
            items: list[dict[str, Any]] = []
            for row in rows:
                entry_id = int(row["id"])
                meanings = [
                    str(meaning["meaning"])
                    for meaning in connection.execute(
                        """
                        SELECT meaning FROM learning_meanings
                        WHERE entry_id = ? ORDER BY id
                        """,
                        (entry_id,),
                    )
                ]
                items.append(
                    {
                        "entry_id": entry_id,
                        "kind": str(row["kind"]),
                        "lemma": str(row["lemma"]),
                        "display": str(row["display"]),
                        "reading": str(row["reading"]),
                        "meanings": meanings,
                        "encounter_count": int(row["encounter_count"]),
                        "updated_at": str(row["updated_at"]),
                    }
                )
        return items

    def entry_details(self, entry_id: int) -> dict[str, Any] | None:
        """Return one entry and every distinct sentence where it was saved."""

        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, kind, lemma, reading, display, created_at, updated_at
                FROM learning_entries WHERE id = ?
                """,
                (entry_id,),
            ).fetchone()
            if row is None:
                return None
            meanings = [
                str(meaning["meaning"])
                for meaning in connection.execute(
                    """
                    SELECT meaning FROM learning_meanings
                    WHERE entry_id = ? ORDER BY id
                    """,
                    (entry_id,),
                )
            ]
            encounters = [
                {
                    "meaning": str(encounter["meaning"]),
                    "source_url": str(encounter["source_url"]),
                    "job_id": str(encounter["job_id"]),
                    "sentence_id": str(encounter["sentence_id"]),
                    "sentence": str(encounter["sentence"]),
                    "encountered_at": str(encounter["created_at"]),
                }
                for encounter in connection.execute(
                    """
                    SELECT m.meaning, x.source_url, x.job_id, x.sentence_id,
                           x.sentence, x.created_at
                    FROM learning_encounters AS x
                    JOIN learning_meanings AS m ON m.id = x.meaning_id
                    WHERE x.entry_id = ?
                    ORDER BY x.created_at DESC, x.id DESC
                    """,
                    (entry_id,),
                )
            ]
        return {
            "entry_id": int(row["id"]),
            "kind": str(row["kind"]),
            "lemma": str(row["lemma"]),
            "display": str(row["display"]),
            "reading": str(row["reading"]),
            "meanings": meanings,
            "encounter_count": len(encounters),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
            "encounters": encounters,
        }

    def delete_entry(self, entry_id: int) -> bool:
        """Delete one library entry and its forms, meanings, and saved encounters."""

        if entry_id < 1:
            raise ValueError("词库条目 ID 无效")
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute("DELETE FROM learning_entries WHERE id = ?", (entry_id,))
            connection.commit()
            return cursor.rowcount > 0

    def _initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._write_lock, self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                CREATE TABLE IF NOT EXISTS learning_entries (
                    id INTEGER PRIMARY KEY,
                    kind TEXT NOT NULL CHECK(kind IN ('word', 'collocation', 'grammar')),
                    lemma TEXT NOT NULL,
                    reading TEXT NOT NULL DEFAULT '',
                    display TEXT NOT NULL,
                    normalized_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS learning_forms (
                    id INTEGER PRIMARY KEY,
                    entry_id INTEGER NOT NULL REFERENCES learning_entries(id) ON DELETE CASCADE,
                    surface TEXT NOT NULL,
                    reading TEXT NOT NULL DEFAULT '',
                    normalized_surface TEXT NOT NULL,
                    UNIQUE(entry_id, normalized_surface)
                );
                CREATE TABLE IF NOT EXISTS learning_meanings (
                    id INTEGER PRIMARY KEY,
                    entry_id INTEGER NOT NULL REFERENCES learning_entries(id) ON DELETE CASCADE,
                    meaning TEXT NOT NULL,
                    normalized_meaning TEXT NOT NULL,
                    first_context TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(entry_id, normalized_meaning)
                );
                CREATE TABLE IF NOT EXISTS learning_encounters (
                    id INTEGER PRIMARY KEY,
                    entry_id INTEGER NOT NULL REFERENCES learning_entries(id) ON DELETE CASCADE,
                    meaning_id INTEGER NOT NULL REFERENCES learning_meanings(id) ON DELETE CASCADE,
                    context_key TEXT NOT NULL,
                    source_url TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    sentence_id TEXT NOT NULL,
                    sentence TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(entry_id, context_key)
                );
                CREATE INDEX IF NOT EXISTS idx_learning_encounters_entry
                    ON learning_encounters(entry_id);
                CREATE INDEX IF NOT EXISTS idx_learning_meanings_entry
                    ON learning_meanings(entry_id);
                """
            )
            entries = connection.execute(
                "SELECT id, lemma, display FROM learning_entries"
            ).fetchall()
            for entry in entries:
                if _KANJI.search(str(entry["display"])):
                    continue
                form = connection.execute(
                    """
                    SELECT surface FROM learning_forms
                    WHERE entry_id = ? ORDER BY id
                    """,
                    (int(entry["id"]),),
                ).fetchall()
                display = next(
                    (
                        str(candidate["surface"])
                        for candidate in form
                        if _KANJI.search(str(candidate["surface"]))
                    ),
                    str(entry["lemma"]),
                )
                connection.execute(
                    "UPDATE learning_entries SET display = ? WHERE id = ?",
                    (display, int(entry["id"])),
                )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @staticmethod
    def _validate_kind(kind: str) -> LearningItemKind:
        if kind not in _VALID_KINDS:
            raise ValueError("学习词库类型必须是 word、collocation 或 grammar")
        return kind  # type: ignore[return-value]

    @staticmethod
    def _entry_key(kind: LearningItemKind, lemma: str, reading: str) -> str:
        clean_lemma = _normalize_term(lemma)
        clean_reading = _normalize_term(reading)
        if not clean_lemma:
            raise ValueError("学习词库原型不能为空")
        return f"{kind}\u241f{clean_lemma}\u241f{clean_reading}"

    @staticmethod
    def _state_with_connection(
        connection: sqlite3.Connection,
        *,
        key: str,
        meaning: str,
        context_key: str,
    ) -> dict[str, Any]:
        row = connection.execute(
            """
            SELECT e.id,
                   COUNT(DISTINCT m.id) AS meaning_count,
                   COUNT(DISTINCT x.id) AS encounter_count
            FROM learning_entries AS e
            LEFT JOIN learning_meanings AS m ON m.entry_id = e.id
            LEFT JOIN learning_encounters AS x ON x.entry_id = e.id
            WHERE e.normalized_key = ?
            GROUP BY e.id
            """,
            (key,),
        ).fetchone()
        if row is None:
            return {
                "entry_id": None,
                "exists": False,
                "meaning_saved": False,
                "context_saved": False,
                "meaning_count": 0,
                "encounter_count": 0,
            }
        entry_id = int(row["id"])
        meaning_saved = (
            connection.execute(
                """
                SELECT 1 FROM learning_meanings
                WHERE entry_id = ? AND normalized_meaning = ?
                """,
                (entry_id, _normalize_meaning(meaning)),
            ).fetchone()
            is not None
        )
        context_saved = (
            connection.execute(
                """
                SELECT 1 FROM learning_encounters
                WHERE entry_id = ? AND context_key = ?
                """,
                (entry_id, context_key),
            ).fetchone()
            is not None
        )
        return {
            "entry_id": entry_id,
            "exists": True,
            "meaning_saved": meaning_saved,
            "context_saved": context_saved,
            "meaning_count": int(row["meaning_count"]),
            "encounter_count": int(row["encounter_count"]),
        }
