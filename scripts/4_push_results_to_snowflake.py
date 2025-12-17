#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
4_push_results_to_snowflake.py

Objectif :
- Lire un fichier results.json produit par Soda
- Mapper les checks vers T_TESTCASE (via TST_IDF dans le nom du test)
- Consolider les résultats (priorité OK > KO > NOT_EVALUATED > ERROR)
- Insérer dans SNOWFLAKE_SCHEMA.T_TESTCASE_RESULT
"""

import os
import sys
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from qdd_utils import get_logger, connect_snowflake_from_env, env

logger = get_logger("étape 4 : Sauvgarder les résultats dans Snowflake")

DEFAULT_RESULTS_JSON = Path("target/results.json")


STATE_PRIORITY = {
    "OK": 4,
    "KO": 3,
    "NOT_EVALUATED": 2,
    "ERROR": 1,
}


# ---------------------------------------------------------------------------
# LECTURE JSON
# ---------------------------------------------------------------------------

def load_results_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier results.json introuvable : {path}")
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    logger.info("results.json chargé depuis %s", path)
    return data


def parse_timestamp(data: Dict[str, Any]) -> datetime:
    """
    Essaie de récupérer un timestamp d'exécution à partir des champs Soda :
    scanEndTimestamp, scanStartTimestamp, dataTimestamp.
    """
    candidates = [
        data.get("scanEndTimestamp"),
        data.get("scanStartTimestamp"),
        data.get("dataTimestamp"),
    ]
    for val in candidates:
        if not val:
            continue
        if isinstance(val, (int, float)):
            # epoch (s ou ms)
            if val > 1e12:
                return datetime.utcfromtimestamp(val / 1000.0)
            return datetime.utcfromtimestamp(val)
        if isinstance(val, str):
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            except Exception:
                continue
    return datetime.utcnow()


def extract_numeric_testcase_id(label: str) -> Optional[int]:
    """
    Exemples de labels :
      - CRIT-SCHEMA-TABLE-COL-TSTID-0
      - 49.snowflake.user_defined_query[...] (dans certains cas)
      - '95' (juste l'ID)
    On essaie de retrouver un entier.
    """
    if not label:
        return None

    # Si c'est directement un entier
    if label.isdigit():
        return int(label)

    # Cherche pattern '-<id>-0'
    m = re.search(r"-([0-9]+)-0$", label)
    if m:
        return int(m.group(1))

    # Cherche 'TST_ID=<id>' si jamais
    m = re.search(r"TST_ID=([0-9]+)", label)
    if m:
        return int(m.group(1))

    # Dernier fallback : nombre à la fin
    m = re.search(r"([0-9]+)$", label)
    if m:
        return int(m.group(1))

    return None


# ---------------------------------------------------------------------------
# TRANSFORMATION VERS LIGNES T_TESTCASE_RESULT
# ---------------------------------------------------------------------------

def build_result_rows(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    scan_ts = parse_timestamp(data)
    res_rows: List[Dict[str, Any]] = []

    # Mapping des statuts Soda -> statuts QDD
    OUTCOME_MAP = {
        "PASS": "OK",
        "FAIL": "KO",
        "WARN": "KO",            # à adapter si tu veux un traitement différent
        "SKIPPED": "NOT_EVALUATED",
        "ERROR": "ERROR",

        # On accepte aussi déjà le format QDD au cas où
        "OK": "OK",
        "KO": "KO",
        "NOT_EVALUATED": "NOT_EVALUATED",
    }

    checks = data.get("checks", []) or []
    for chk in checks:
        name = chk.get("name") or chk.get("identity") or ""

        outcome_raw = (chk.get("outcome") or "").upper()
        outcome = OUTCOME_MAP.get(outcome_raw, "ERROR")  # défaut = ERROR

        diagnostics = chk.get("diagnostics") or {}

        tst_id = extract_numeric_testcase_id(name)
        if tst_id is None:
            logger.warning("Impossible d'extraire TST_IDF depuis le nom de check : %s", name)
            continue

        # Valeur mesurée
        value = diagnostics.get("value")
        if value is None and "metrics" in diagnostics:
            for k, v in diagnostics["metrics"].items():
                if isinstance(v, (int, float)):
                    value = v
                    break

        # Optionnel : si tu veux que ERROR / NOT_EVALUATED aient valeur NULL
        if outcome in ("ERROR", "NOT_EVALUATED"):
            value = None

        res_rows.append(
            {
                "RES_ID_TESTCASE": tst_id,
                "RES_DATE_EXECUTION": scan_ts,
                "RES_VALEUR_MESUREE": value,
                "RES_RESULTAT": outcome,
            }
        )

    logger.info("%d lignes brutes de résultats Soda", len(res_rows))
    return res_rows



def deduplicate_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    On garde 1 ligne par RES_ID_TESTCASE avec la priorité de résultat :
    OK > KO > NOT_EVALUATED > ERROR.
    """
    best_by_id: Dict[int, Dict[str, Any]] = {}

    for r in rows:
        tid = r["RES_ID_TESTCASE"]
        state = r["RES_RESULTAT"]
        prio = STATE_PRIORITY.get(state, 0)

        if tid not in best_by_id:
            best_by_id[tid] = r
            continue

        current_state = best_by_id[tid]["RES_RESULTAT"]
        current_prio = STATE_PRIORITY.get(current_state, 0)
        if prio > current_prio:
            best_by_id[tid] = r

    dedup = list(best_by_id.values())
    logger.info("Résultats dédupliqués : %d lignes (sur %d)", len(dedup), len(rows))
    return dedup


# ---------------------------------------------------------------------------
# INSERTION SNOWFLAKE
# ---------------------------------------------------------------------------

def insert_results(conn, rows: List[Dict[str, Any]], table_name: str) -> None:
    if not rows:
        logger.info("Aucun résultat à insérer dans %s.", table_name)
        return

    sql = f"""
        INSERT INTO {table_name} (
            RES_ID_TESTCASE,
            RES_DATE_EXECUTION,
            RES_VALEUR_MESUREE,
            RES_RESULTAT
        )
        VALUES (%s, %s, %s, %s)
    """

    cur = conn.cursor()
    try:
        payload = [
            (
                r["RES_ID_TESTCASE"],
                r["RES_DATE_EXECUTION"],
                r["RES_VALEUR_MESUREE"],
                r["RES_RESULTAT"],
            )
            for r in rows
        ]
        cur.executemany(sql, payload)
        conn.commit()
        logger.info("%d résultats insérés dans %s", len(rows), table_name)
    finally:
        cur.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    # Fichier JSON (param 1 éventuel, sinon default)
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        json_path = DEFAULT_RESULTS_JSON

    data = load_results_json(json_path)
    raw_rows = build_result_rows(data)
    rows = deduplicate_results(raw_rows)

    conn = connect_snowflake_from_env(role_env="SNOWFLAKE_ROLE")

    try:
        schema = env("SNOWFLAKE_SCHEMA", required=True)
        table_name = os.getenv("T_TESTCASE_RESULT_TABLE", "T_TESTCASE_RESULT")
        full_table = f"{schema}.{table_name}"
        insert_results(conn, rows, full_table)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
