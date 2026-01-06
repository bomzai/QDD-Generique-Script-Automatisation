#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
from collections import defaultdict
import os
import re
import yaml
from typing import Dict, Any, List, Tuple, Optional

from qdd_utils import (
    get_logger,
    connect_snowflake_from_env,
    classify_metric,
    MANIFEST_PATH,
    SODA_CHECKS_DIR,
    DBML_NAME,
    get_target_dbml_schema,
    DBML_PATH,
    load_manifest,
    ensure_dbml_version,
    TABLE_TESTCASE,
    TABLE_METRIQUE,
    
)
from qdd_utils import q_ident, q_qualified, q_str, write_yaml_file, SODA_CHECKS_DIR

logger = get_logger("étape 2 : génerer les checks SODA")

OUTPUT_DIR = SODA_CHECKS_DIR


# -----------------------------------------------------------------------------
# Point 1: parse_integrity_ref robuste (regex tolérante)
# -----------------------------------------------------------------------------
_INTEGRITY_RE = re.compile(
    r"\bSUR\s+([A-Z0-9_]+)\.([A-Z0-9_]+)\.([A-Z0-9_]+)\s+DE\s+R[ÉE]F[ÉE]RENCE\b",
    re.IGNORECASE,
)

def parse_integrity_ref(description: str) -> Optional[Tuple[str, str]]:
    """
    Cherche dans TST_DESCRIPTION un pattern du type :

      '... sur PERSONNE.T_ORGANISATION.PER_IDF de référence'
      '... sur LOV.T_T_PROD.CODE de référence'
      '... sur PRODUIT.T_PRODUIT.PRD_CODE_PRODUIT de réference'

    Retourne (table_logique, colonne) = ('PERSONNE.T_ORGANISATION', 'PER_IDF')
    ou None si non trouvé.
    """
    if not description:
        return None

    m = _INTEGRITY_RE.search(description)
    if not m:
        return None

    schema, table, col = m.group(1), m.group(2), m.group(3)
    return f"{schema.upper()}.{table.upper()}", col.upper()

def parse_validation_rule(description: str) -> Optional[str]:
    """
    Extrait le SQL de règle depuis TST_DESCRIPTION pour les tests validation_metier.

    Pattern: "... - RULE_SQL: <expression_sql>"

    Example:
        Input: "Test validation - RULE_SQL: {column} < 0"
        Output: "{column} < 0"

    Returns:
        Expression SQL ou None si non trouvé
    """
    if not description:
        return None

    marker = "RULE_SQL:"
    idx = description.find(marker)
    if idx == -1:
        return None

    rule_sql = description[idx + len(marker):].strip()
    return rule_sql if rule_sql else None

# -----------------------------------------------------------------------------
# Mapping logique -> physique (inchangé)
# -----------------------------------------------------------------------------
def dataset_name_from_table(table_cible: str) -> str:
    """
    TST_TABLE_CIBLE = 'PRODUIT.T_PRODUIT'
    => dataset Soda = 'SCH_REF_PRODUIT.T_PRODUIT'
    (schéma hardcodé SCH_REF_<SCHEMA> ici).
    """
    if not table_cible:
        return ""
    table_cible = table_cible.strip()
    if "." not in table_cible:
        return table_cible.upper()
    schema, table = table_cible.split(".", 1)
    schema_up = schema.strip().upper()
    table_up = table.strip().upper()
    return f"{schema_up}.{table_up}"


def ref_dataset_name_from_table(ref_table: str) -> str:
    """
    Transforme une table logique 'SCHEMA.T_TABLE' en table physique 'SCH_REF_SCHEMA.T_TABLE'.
    """
    if not ref_table or "." not in ref_table:
        return ref_table or ""

    schema, table = ref_table.split(".", 1)
    schema_up = schema.strip().upper()
    table_up = table.strip().upper()

    return f"{schema_up}.{table_up}"


def get_target_referential_schema() -> str:
    """
    Renvoie le schéma logique du référentiel ciblé .
    - d'abord via get_target_dbml_schema(manifest, dbml_name)
    - sinon DBML_NAME
    - sinon 'PRODUIT'
    """
    schema = get_target_dbml_schema(MANIFEST_PATH, DBML_NAME)
    if schema:
        return schema.upper()
    return (DBML_NAME or "CUSTOMER").upper()

# -----------------------------------------------------------------------------
# DBML VERSION (Commit 1)
# - Objectif : ne générer/exécuter que les testcases de la version DBML courante
# -----------------------------------------------------------------------------

def extract_project_meta_from_dbml_file(dbml_path: Path) -> Dict[str, Optional[str]]:
    """Lit le DBML brut pour extraire Project { name, version }."""
    p = Path(dbml_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Fichier DBML introuvable : {p}. "
            "Assure-toi qu'il est généré (ex: via 0_fetch_dbml.py) avant d'exécuter l'étape 2."
        )

    content = p.read_text(encoding="utf-8")

    project_name: Optional[str] = None
    project_version: Optional[str] = None

    m = re.search(r"Project\s+(\w+)\s*\{([^}]*)\}", content, flags=re.DOTALL | re.IGNORECASE)
    if m:
        project_block = m.group(2)
        m_name = re.search(r'name:\s*"([^"]+)"', project_block)
        m_ver = re.search(r'version:\s*"([^"]+)"', project_block)
        if m_name:
            project_name = m_name.group(1).strip()
        if m_ver:
            project_version = m_ver.group(1).strip()

    return {"project_name": project_name, "project_version": project_version}


def get_dbml_entry_from_manifest(
    manifest: Dict[str, Any],
    project_name: str,
    fallback_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Récupère l'entrée manifest['dbml'][] la plus pertinente."""
    entries = manifest.get("dbml", []) or []

    for entry in entries:
        if entry.get("name") == project_name:
            return entry

    if fallback_name:
        for entry in entries:
            if entry.get("name") == fallback_name:
                return entry

    return entries[0] if entries else {}


def get_current_dbml_version_id(conn) -> int:
    """Calcule le DBV_IDF correspondant au DBML courant (Project name/version + schema_cible)."""
    manifest = load_manifest(str(MANIFEST_PATH))
    meta = extract_project_meta_from_dbml_file(DBML_PATH)

    project_name = meta.get("project_name") or (manifest.get("project", {}) or {}).get("name") or (DBML_NAME or "Customer")
    project_version = meta.get("project_version") or (manifest.get("project", {}) or {}).get("version") or "0.0.0"

    dbml_entry = get_dbml_entry_from_manifest(manifest, project_name=str(project_name), fallback_name=DBML_NAME)
    dbml_version_id = ensure_dbml_version(conn, str(project_name), str(project_version), dbml_entry, logger=logger)

    logger.info(
        "DBML courante => DBV_IDF=%s (projet=%s, version=%s, schema_cible=%s)",
        dbml_version_id,
        project_name,
        project_version,
        dbml_entry.get("schema", ""),
    )
    return int(dbml_version_id)



# -----------------------------------------------------------------------------
# Récupération des testcases
# -----------------------------------------------------------------------------
def load_testcases_with_metrics(conn) -> List[Dict[str, Any]]:
    """    Joint T_TESTCASE et T_METRIQUE, retourne une liste de dicts.

    Filtres :
      - référentiel courant (schéma logique) : TST_TABLE_CIBLE LIKE '<SCHEMA>.%'
      - TST_TYPE = 'AUTO_GENERER'
      - validité : TST_VALIDE_JUSQUA IS NULL (pas de fin) OU > CURRENT_DATE()
      - version DBML courante : TST_ID_DBML_VERSION = <DBV_IDF>

    Guard legacy :
      - si aucun testcase n'est versionné mais qu'il existe des lignes legacy avec TST_ID_DBML_VERSION IS NULL,
        on peut (optionnellement) retomber dessus (contrôlé par QDD_ALLOW_LEGACY_NULL_DBML_VERSION, défaut: true).

    Dédup : garde la ligne la plus récente par (nom + table + champ + métrique)
    """
    #target_schema = get_target_referential_schema()
    #like_pattern = f"{target_schema}.%"

    #dbml_version_id = get_current_dbml_version_id(conn)

    # Extraction des métadonnées REELLES du fichier DBML
    meta = extract_project_meta_from_dbml_file(DBML_PATH)
    
    manifest = load_manifest(str(MANIFEST_PATH))
    project_name = meta.get("project_name") or (manifest.get("project") or {}).get("name")
    project_version = meta.get("project_version") or (manifest.get("project") or {}).get("version")
    
    # Récupération de l'entrée manifest pour avoir le schéma cible exact
    dbml_entry = get_dbml_entry_from_manifest(manifest, project_name=str(project_name), fallback_name=DBML_NAME)
    
    # On utilise le schéma du manifest (ex: CUSTOMER_SCHEMA)

    target_schema = dbml_entry.get("schema", "PRODUIT").upper()
    like_pattern = f"{target_schema}.%"

    # 3. Récupération de l'ID de version (sera identique au Script 1 : ID=1)
    dbml_version_id = ensure_dbml_version(conn, str(project_name), str(project_version), dbml_entry, logger=logger)

    logger.info("Recherche tests pour Schéma: %s | DBV_IDF: %s", like_pattern, dbml_version_id)

    allow_legacy = os.getenv("QDD_ALLOW_LEGACY_NULL_DBML_VERSION", "true").strip().lower() in (
        "1", "true", "yes", "y"
    )

    sql_versioned = f"""
        SELECT
            TST_IDF,
            TST_NOM_TEST,
            TST_TABLE_CIBLE,
            TST_CHAMP_CIBLE,
            TST_SEUIL_BORNE_INFERIEURE,
            TST_SEUIL_BORNE_SUPERIEURE,
            TST_DESCRIPTION,
            TST_ID_DBML_VERSION,
            MET_NOM_METRIQUE,
            MET_TYPE
        FROM (
            SELECT
                T.TST_IDF,
                T.TST_NOM_TEST,
                T.TST_TABLE_CIBLE,
                T.TST_CHAMP_CIBLE,
                T.TST_SEUIL_BORNE_INFERIEURE,
                T.TST_SEUIL_BORNE_SUPERIEURE,
                T.TST_DESCRIPTION,
                T.TST_ID_DBML_VERSION,
                T.TST_DATE_CREATION,
                M.MET_NOM_METRIQUE,
                M.MET_TYPE,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        T.TST_NOM_TEST,
                        T.TST_TABLE_CIBLE,
                        T.TST_CHAMP_CIBLE,
                        T.TST_IDF_METRIQUE
                    ORDER BY T.TST_DATE_CREATION DESC
                ) AS RN
            FROM {TABLE_TESTCASE} T
            JOIN {TABLE_METRIQUE} M
              ON T.TST_IDF_METRIQUE = M.MET_IDF
            WHERE
                T.TST_TYPE = 'AUTO_GENERER'
                AND (T.TST_VALIDE_JUSQUA IS NULL OR T.TST_VALIDE_JUSQUA > CURRENT_DATE())
                AND T.TST_TABLE_CIBLE LIKE %s
                AND T.TST_ID_DBML_VERSION = %s
        )
        WHERE RN = 1
    """

    sql_legacy_null = f"""
        SELECT
            TST_IDF,
            TST_NOM_TEST,
            TST_TABLE_CIBLE,
            TST_CHAMP_CIBLE,
            TST_SEUIL_BORNE_INFERIEURE,
            TST_SEUIL_BORNE_SUPERIEURE,
            TST_DESCRIPTION,
            TST_ID_DBML_VERSION,
            MET_NOM_METRIQUE,
            MET_TYPE
        FROM (
            SELECT
                T.TST_IDF,
                T.TST_NOM_TEST,
                T.TST_TABLE_CIBLE,
                T.TST_CHAMP_CIBLE,
                T.TST_SEUIL_BORNE_INFERIEURE,
                T.TST_SEUIL_BORNE_SUPERIEURE,
                T.TST_DESCRIPTION,
                T.TST_ID_DBML_VERSION,
                T.TST_DATE_CREATION,
                M.MET_NOM_METRIQUE,
                M.MET_TYPE,
                ROW_NUMBER() OVER (
                    PARTITION BY
                        T.TST_NOM_TEST,
                        T.TST_TABLE_CIBLE,
                        T.TST_CHAMP_CIBLE,
                        T.TST_IDF_METRIQUE
                    ORDER BY T.TST_DATE_CREATION DESC
                ) AS RN
            FROM {TABLE_TESTCASE} T
            JOIN {TABLE_METRIQUE} M
              ON T.TST_IDF_METRIQUE = M.MET_IDF
            WHERE
                T.TST_TYPE = 'AUTO_GENERER'
                AND (T.TST_VALIDE_JUSQUA IS NULL OR T.TST_VALIDE_JUSQUA > CURRENT_DATE())
                AND T.TST_TABLE_CIBLE LIKE %s
                AND T.TST_ID_DBML_VERSION IS NULL
        )
        WHERE RN = 1
    """

    sql_count_legacy = f"""
        SELECT COUNT(*)
        FROM {TABLE_TESTCASE} T
        WHERE
            T.TST_TYPE = 'AUTO_GENERER'
            AND (T.TST_VALIDE_JUSQUA IS NULL OR T.TST_VALIDE_JUSQUA > CURRENT_DATE())
            AND T.TST_TABLE_CIBLE LIKE %s
            AND T.TST_ID_DBML_VERSION IS NULL
    """

    cur = conn.cursor()
    try:
        # 1) Version DBML courante (prioritaire)
        cur.execute(sql_versioned, (like_pattern, dbml_version_id))
        rows = cur.fetchall()
        cols = [c[0] for c in cur.description]

        # 2) Guard legacy (NULL) si rien trouvé
        if not rows:
            cur.execute(sql_count_legacy, (like_pattern,))
            legacy_count = int((cur.fetchone() or [0])[0] or 0)

            if legacy_count > 0:
                msg = (
                    f"Aucun testcase trouvé pour DBV_IDF={dbml_version_id} (DBML courante), "
                    f"mais {legacy_count} testcases legacy ont TST_ID_DBML_VERSION=NULL."
                )

                if not allow_legacy:
                    raise RuntimeError(
                        msg + "\nRelance d'abord 1_generate_testcases.py pour regenerer des testcases versionnes."
                    )

                logger.warning(
                    msg
                    + "\nFallback : execution des testcases legacy (NULL) car QDD_ALLOW_LEGACY_NULL_DBML_VERSION=true."
                )

                cur.execute(sql_legacy_null, (like_pattern,))
                rows = cur.fetchall()
                cols = [c[0] for c in cur.description]

        results: List[Dict[str, Any]] = [dict(zip(cols, r)) for r in rows]
        logger.info("%d testcases chargés pour génération Soda.", len(results))
        return results
    finally:
        cur.close()



# -----------------------------------------------------------------------------
# Génération Soda checks
# -----------------------------------------------------------------------------
def generate_soda_checks_for_table(dataset: str, testcases: List[Dict[str, Any]]) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []

    for tc in testcases:
        met_name = tc["MET_NOM_METRIQUE"]
        critere = tc["MET_TYPE"]
        kind = classify_metric(met_name, critere)

        col = (tc["TST_CHAMP_CIBLE"] or "").strip()
        table_cible = (tc["TST_TABLE_CIBLE"] or "").strip()
        desc = tc.get("TST_DESCRIPTION") or ""
        seuil_inf = tc.get("TST_SEUIL_BORNE_INFERIEURE")
        seuil_sup = tc.get("TST_SEUIL_BORNE_SUPERIEURE")

        # Point 3: fail-fast si seuils absents (même si normalement NOT NULL côté modèle)
        if seuil_inf is None or seuil_sup is None:
            logger.error(
                "Seuils NULL pour testcase %s (%s.%s) => génération Soda stoppée",
                tc.get("TST_IDF"), tc.get("TST_TABLE_CIBLE"), tc.get("TST_CHAMP_CIBLE")
            )
            raise SystemExit(1)

        low = float(seuil_inf)
        high = float(seuil_sup)

        attributes = {
            "description": desc,
            "tags": "QDD,TECH",
            "TST_IDF": tc["TST_IDF"],
        }

        # ---------------------------------------------------------
        # COMPLETUDE
        # ---------------------------------------------------------
        if kind == "completude":
            if not col:
                logger.warning("Testcase %s sans colonne cible pour complétude, ignoré.", tc["TST_IDF"])
                continue

            header = f"missing_percent({col}) between {low} and {high}"
            body = {"name": tc["TST_NOM_TEST"], "attributes": attributes}
            checks.append({header: body})
            continue

        # ---------------------------------------------------------
        # UNICITE
        # ---------------------------------------------------------
        if kind == "unicite":
            if not col:
                logger.warning("Testcase %s sans colonne cible pour unicité, ignoré.", tc["TST_IDF"])
                continue

            cols = [c.strip() for c in col.split(",") if c.strip()]
            if not cols:
                logger.warning("Testcase %s avec liste de colonnes vide pour unicité, ignoré.", tc["TST_IDF"])
                continue

            col_expr = cols[0] if len(cols) == 1 else ", ".join(cols)
            header = f"duplicate_percent({col_expr}) between {low} and {high}"
            body = {"name": tc["TST_NOM_TEST"], "attributes": attributes}
            checks.append({header: body})
            continue

        # ---------------------------------------------------------
        # INTEGRITE (Point 2 + Point 6 : quoting + NOT EXISTS)
        # ---------------------------------------------------------
        if kind == "integrite":
            ref_info = parse_integrity_ref(desc)
            if not ref_info:
                logger.warning(
                    "Impossible d'extraire la table de référence pour le testcase %s (desc tronquée: '%s')",
                    tc["TST_IDF"], desc[:200],
                )
                continue

            if not col:
                logger.warning("Testcase %s sans colonne cible pour intégrité, ignoré.", tc["TST_IDF"])
                continue

            ref_table_logical, ref_col = ref_info
            ref_table_physical = ref_dataset_name_from_table(ref_table_logical)

            metric_name = f"invalid_count-int-{table_cible}-{col}"
            header = f"{metric_name} between {low} and {high}"

            # Quoting + validation
            dataset_sql = q_qualified(dataset)
            col_sql = q_ident(col)
            ref_table_sql = q_qualified(ref_table_physical)
            ref_col_sql = q_ident(ref_col)

            query_sql = (
                f'SELECT COUNT(*) AS "{metric_name}"\n'
                f"FROM {dataset_sql} t\n"
                f"WHERE t.{col_sql} IS NOT NULL\n"
                f"  AND NOT EXISTS (\n"
                f"    SELECT 1\n"
                f"    FROM {ref_table_sql} r\n"
                f"    WHERE r.{ref_col_sql} = t.{col_sql}\n"
                f"  )"
            )

            body = {
                "name": tc["TST_NOM_TEST"],
                f"{metric_name} query": query_sql,
                "attributes": attributes,
            }
            checks.append({header: body})
            continue

        # ---------------------------------------------------------
        # TRACABILITE (Point 4 + Point 2 : sentinelle + quoting/validation)
        # ---------------------------------------------------------
        if kind == "tracabilite":
            if not col:
                logger.warning("Testcase %s sans colonne cible pour traçabilité, ignoré.", tc["TST_IDF"])
                continue

            if "." in dataset:
                schema_for_query, table_for_query = dataset.split(".", 1)
                schema_for_query = schema_for_query.strip()
                table_for_query = table_for_query.strip()
            else:
                logger.warning(
                    "Dataset '%s' inattendu pour traçabilité (pas de '.'). Testcase %s ignoré.",
                    dataset, tc["TST_IDF"],
                )
                continue

            # Validation "identifiants" (bloque noms bizarres)
            _ = q_ident(schema_for_query)
            _ = q_ident(table_for_query)
            _ = q_ident(col)

            metric_name = "traceability_ratio"
            header = f"{metric_name} between {low} and {high}"

            # Sentinelle hors-plage => KO garanti sans erreur SQL
            sentinel = float(high) + 1.0

            query_sql = (
                "SELECT\n"
                "  CASE\n"
                f"    WHEN COUNT(*) = 0 THEN {sentinel}  -- colonne introuvable => KO (hors plage)\n"
                "    ELSE SUM(CASE WHEN COMMENT IS NULL OR COMMENT = '' THEN 1 ELSE 0 END) * 1.0 / COUNT(*)\n"
                "  END AS traceability_ratio\n"
                "FROM INFORMATION_SCHEMA.COLUMNS\n"
                f"WHERE TABLE_SCHEMA = {q_str(schema_for_query.upper())}\n"
                f"  AND TABLE_NAME   = {q_str(table_for_query.upper())}\n"
                f"  AND COLUMN_NAME  = {q_str(col.upper())}"
            )

            body = {
                "name": tc["TST_NOM_TEST"],
                f"{metric_name} query": query_sql,
                "attributes": attributes,
            }
            checks.append({header: body})
            continue
        
        # ---------------------------------------------------------
        # VALIDATION MÉTIER (Business Validations)
        # ---------------------------------------------------------
        if kind == "validation_metier":
            if not col:
                logger.warning("Testcase %s sans colonne pour validation métier, ignoré.", tc["TST_IDF"])
                continue

            # Parser le SQL de règle depuis la description
            rule_sql = parse_validation_rule(desc)
            if not rule_sql:
                logger.error(
                    "Impossible d'extraire RULE_SQL pour testcase %s (desc: '%s'). "
                    "Format attendu: '... - RULE_SQL: <expression>'",
                    tc["TST_IDF"], desc[:200]
                )
                continue

            # Remplacer {column} par nom de colonne qualifié
            col_sql = q_ident(col)
            rule_sql_expanded = rule_sql.replace("{column}", f"t.{col_sql}")

            # Nom de métrique et header
            metric_name = f"invalid_count-val-{table_cible}-{col}"
            header = f"{metric_name} between {low} and {high}"

            # Requête SQL comptant les violations
            dataset_sql = q_qualified(dataset)

            query_sql = (
                f'SELECT COUNT(*) AS "{metric_name}"\n'
                f"FROM {dataset_sql} t\n"
                f"WHERE {rule_sql_expanded}"
            )

            body = {
                "name": tc["TST_NOM_TEST"],
                f"{metric_name} query": query_sql,
                "attributes": attributes,
            }
            checks.append({header: body})
            continue

        # ---------------------------------------------------------
        # AUTRES
        # ---------------------------------------------------------
        logger.info("Testcase %s (metrique '%s') classé en 'autre', ignoré.", tc["TST_IDF"], met_name)

        
    return {f"checks for {dataset}": checks}

import yaml

class _LiteralStr(str):
    """Marqueur pour forcer un YAML block scalar '|'."""
    pass

def _literal_str_representer(dumper: yaml.SafeDumper, data: _LiteralStr):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")

yaml.add_representer(_LiteralStr, _literal_str_representer, Dumper=yaml.SafeDumper)

def _to_literal_multiline(obj):
    """Convertit récursivement les strings multi-lignes en _LiteralStr."""
    if isinstance(obj, str) and "\n" in obj:
        return _LiteralStr(obj)
    if isinstance(obj, dict):
        return {k: _to_literal_multiline(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_literal_multiline(v) for v in obj]
    return obj


# -----------------------------------------------------------------------------
# YAML writer
# -----------------------------------------------------------------------------
def write_soda_yaml(dataset: str, yaml_dict: Dict[str, Any]):
    """
    Écrit le YAML SodaCL dans OUTPUT_DIR/<dataset>.yml
    - Les strings multi-lignes sont écrites en block scalar '|'
    - OUTPUT_DIR reste patchable par les tests (monkeypatch)
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = dataset.replace(".", "_") + ".yml"
    path = OUTPUT_DIR / filename

    payload = _to_literal_multiline(yaml_dict)

    content = yaml.dump(
        payload,
        sort_keys=False,
        allow_unicode=True,
        width=4096,  # évite le wrapping agressif
        Dumper=yaml.SafeDumper,
    )

    path.write_text(content, encoding="utf-8")
    logger.info("Fichier SodaCL généré : %s", path)


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    conn = connect_snowflake_from_env(role_env="SNOWFLAKE_ROLE")

    try:
        rows = load_testcases_with_metrics(conn)

        by_table: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_table[r["TST_TABLE_CIBLE"]].append(r)

        for table_cible, tcs in by_table.items():
            dataset = dataset_name_from_table(table_cible)
            if not dataset:
                logger.warning("Table cible vide pour certains testcases, ignorés.")
                continue

            yaml_dict = generate_soda_checks_for_table(dataset, tcs)
            write_soda_yaml(dataset, yaml_dict)

    finally:
        conn.close()


if __name__ == "__main__":
    main()