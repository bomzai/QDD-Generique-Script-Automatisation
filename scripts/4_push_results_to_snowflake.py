#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
4_push_results_to_snowflake.py

Objectif :
- Lire un fichier results.json produit par Soda
- Mapper les checks vers T_TESTCASE (via TST_IDF présent dans les attributes / resourceAttributes)
- Consolider les résultats (priorité OK > KO > NOT_EVALUATED > ERROR)
- Insérer dans SNOWFLAKE_SCHEMA.T_TESTCASE_RESULT
"""

import os
import sys
import json
import re
from qdd_utils import q_qualified
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from qdd_utils import get_logger, connect_snowflake_from_env, env

logger = get_logger("étape 4 : Sauvgarder les résultats dans Snowflake")

DEFAULT_RESULTS_JSON = Path("results.json")


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
    Ancienne logique basée sur un ID numérique dans le nom du test.

    Exemples de labels :
      - CRIT-SCHEMA-TABLE-COL-TSTID-0
      - 49.snowflake.user_defined_query[.] (dans certains cas)
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


def extract_tst_id_from_check(chk: Dict[str, Any]) -> Optional[str]:
    """
    Nouvelle logique : on récupère TST_IDF tel qu'il a été propagé dans les checks Soda.

    Ordre de recherche :
    1) resourceAttributes : liste de { "name": ..., "value": ... }
    2) attributes (dict direct, si présent)
    3) regex dans definition (fallback, en lisant le YAML texte)
    """
    # 1) resourceAttributes
    ra = chk.get("resourceAttributes") or []
    for item in ra:
        if not isinstance(item, dict):
            continue
        if item.get("name") == "TST_IDF":
            val = item.get("value")
            if val:
                return str(val)

    # 2) attributes (au cas où Soda remonte un dict "attributes")
    attrs = chk.get("attributes") or {}
    if isinstance(attrs, dict):
        val = attrs.get("TST_IDF")
        if val:
            return str(val)

    # 3) fallback : parse du YAML "definition" (TST_IDF: "<valeur>")
    definition = chk.get("definition")
    if isinstance(definition, str):
        m = re.search(r'TST_IDF:\s*"([^"]+)"', definition)
        if m:
            return m.group(1)

    logger.warning(
        "Impossible de retrouver TST_IDF dans les attributs pour le check : %s",
        chk.get("name"),
    )
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

        # ---- NOUVEAU : on récupère le TST_IDF (VARCHAR) depuis les attributes ----
        tst_id = extract_tst_id_from_check(chk)
        #if not tst_id:
            # Si tu veux garder un fallback ultra-dégradé basé sur l'ancien parsing,
            # tu peux décommenter les lignes ci-dessous :
            #
            # numeric_id = extract_numeric_testcase_id(name)
            # if numeric_id is not None:
            #     tst_id = str(numeric_id)
            # else:
            #     continue
        #    continue
        if not tst_id:
            logger.error(
        "Check sans TST_IDF ignoré (name=%s, outcome=%s)",
            chk.get("name"),
            chk.get("outcome"),
            )
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
                "RES_ID_TESTCASE": tst_id,      # VARCHAR(50) => TST_IDF
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
    best_by_id: Dict[str, Dict[str, Any]] = {}

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

    merge_sql = f"""
        MERGE INTO {table_name} t
        USING (
            SELECT
                %s::VARCHAR      AS RES_ID_TESTCASE,
                %s::TIMESTAMP_NTZ AS RES_DATE_EXECUTION,
                %s::FLOAT        AS RES_VALEUR_MESUREE,
                %s::VARCHAR      AS RES_RESULTAT
        ) s
        ON  t.RES_ID_TESTCASE   = s.RES_ID_TESTCASE
        AND t.RES_DATE_EXECUTION = s.RES_DATE_EXECUTION
        WHEN MATCHED THEN UPDATE SET
            RES_VALEUR_MESUREE = s.RES_VALEUR_MESUREE,
            RES_RESULTAT       = s.RES_RESULTAT
        WHEN NOT MATCHED THEN INSERT (
            RES_ID_TESTCASE,
            RES_DATE_EXECUTION,
            RES_VALEUR_MESUREE,
            RES_RESULTAT
        ) VALUES (
            s.RES_ID_TESTCASE,
            s.RES_DATE_EXECUTION,
            s.RES_VALEUR_MESUREE,
            s.RES_RESULTAT
        )
    """

    payload = [
        (
            r["RES_ID_TESTCASE"],       # ex: "TC_249bea..."
            r["RES_DATE_EXECUTION"],    # datetime ou string ISO
            r["RES_VALEUR_MESUREE"],    # float (ou None)
            r["RES_RESULTAT"],          # ex: "OK"/"KO"/"ERROR" etc.
        )
        for r in rows
    ]

    cur = conn.cursor()
    try:
        cur.executemany(merge_sql, payload)
        conn.commit()
        logger.info("%d résultats upsertés (MERGE) dans %s", len(rows), table_name)
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
        full_table = q_qualified(f"{schema}.{table_name}")
        insert_results(conn, rows, full_table)
    finally:
        conn.close()


if __name__ == "__main__":
    main()