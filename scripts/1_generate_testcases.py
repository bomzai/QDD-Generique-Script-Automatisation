#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional

from qdd_utils import get_logger, connect_snowflake_from_env
from pydbml import PyDBML
import yaml

logger = get_logger("étape 1 : génerer les testcases")


# ---------------------------------------------------------------------------
# PARAMÈTRES QDD
# ---------------------------------------------------------------------------
# Génération des tests
DEFAULT_SOURCE_CIBLE_ID = os.getenv("QDD_DEFAULT_SOURCE_CIBLE_ID", "1")
DEFAULT_POIDS = float(os.getenv("QDD_DEFAULT_POIDS", "1.0"))
DEFAULT_SEUIL_INF = float(os.getenv("QDD_DEFAULT_SEUIL_INF", "0.0"))
DEFAULT_SEUIL_SUP = float(os.getenv("QDD_DEFAULT_SEUIL_SUP", "0.0"))
DEFAULT_FREQ = os.getenv("QDD_DEFAULT_FREQ", "J")

# Validité
DEFAULT_VALIDE_DE = os.getenv("QDD_DEFAULT_VALIDE_DE", "1900-01-01")
DEFAULT_VALIDE_JUSQUA = os.getenv("QDD_DEFAULT_VALIDE_JUSQUA", "2099-12-31")

# Fichier DBML 
DBML_PATH = os.getenv("DBML_PATH", "dbml/customer.dbml")

# manifest.yml
MANIFEST_PATH = os.getenv("MANIFEST_PATH", "manifest.yml")

# Fichier token GitLab
GITLAB_TOKEN_FILE = os.getenv("GITLAB_TOKEN_FILE", "target/api-key")

# Dossier cible pour les scripts SQL
SQL_TARGET_DIR = os.getenv("SQL_TARGET_DIR", "target")

# Contrôle du Push GitLab
PUSH_TO_GITLAB = os.getenv("PUSH_TO_GITLAB", "FALSE").strip().upper() == "FALSE"

# ---------------------------------------------------------------------------
# UTILITAIRES DATES
# ---------------------------------------------------------------------------

def parse_date(s: str) -> date:
    """
    Convertit une chaîne 'YYYY-MM-DD' en date.
    """
    return datetime.strptime(s, "%Y-%m-%d").date()



def compute_default_dates() -> (date, date):
    """
    Calcule les dates par défaut (valide_de / valide_jusqua)
    """
    try:
        d_de = parse_date(DEFAULT_VALIDE_DE)
    except Exception:
        d_de = date(1900, 1, 1)

    try:
        d_jusqua = parse_date(DEFAULT_VALIDE_JUSQUA)
    except Exception:
        d_jusqua = date(2099, 12, 31)

    return d_de, d_jusqua

# ---------------------------------------------------------------------------
# CHARGEMENT DE LA CLÉ PRIVÉE
# ---------------------------------------------------------------------------

def load_private_key(path: str):
    """
    Charge la clé privée Snowflake au format PEM.
    (Toujours présente, même si get_connection utilise désormais qdd_utils.)
    """
    if not path:
        raise RuntimeError("PRIVATE_KEY_PATH non défini dans les variables d'environnement.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Clé privée introuvable : {path}")
    with open(path, "rb") as f:
        key_data = f.read()
    private_key = serialization.load_pem_private_key(key_data, password=None)
    logger.info(f"Clé privée chargée depuis {path}")
    return private_key

# ---------------------------------------------------------------------------
# CONNEXION SNOWFLAKE via qdd_utils
# ---------------------------------------------------------------------------

def get_connection():
    """Connexion Snowflake via qdd_utils.connect_snowflake_from_env."""
    try:
        conn = connect_snowflake_from_env(role_env="SNOWFLAKE_ROLE")
        logger.info("Connexion Snowflake OK.")
        return conn
    except Exception as e:
        logger.error(f"Erreur connexion Snowflake : {e}")
        raise

# ---------------------------------------------------------------------------
# CHARGEMENT MANIFEST & DBML
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str) -> Dict[str, Any]:
    """
    Charge le fichier manifest.yml et renvoie un dict.
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.yml introuvable : {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    logger.info(f"manifest.yml chargé depuis {manifest_path}")
    return data


def load_dbml(dbml_path: str) -> PyDBML:
    """
    Charge le fichier DBML.
    """
    if not os.path.exists(dbml_path):
        raise FileNotFoundError(f"Fichier DBML introuvable : {dbml_path}")
    with open(dbml_path, "r", encoding="utf-8") as f:
        content = f.read()
    logger.info(f"DBML chargé depuis {dbml_path}")
    return PyDBML(content)


def extract_project_meta_from_dbml_file(dbml_path: str) -> Dict[str, Optional[str]]:
    """
    Lit le DBML brut pour extraire les infos Project { name, version }.
    """
    if not os.path.exists(dbml_path):
        raise FileNotFoundError(f"Fichier DBML introuvable : {dbml_path}")
    with open(dbml_path, "r", encoding="utf-8") as f:
        content = f.read()

    project_name = None
    project_version = None

    m = re.search(
        r'Project\s+(\w+)\s*\{([^}]*)\}',
        content,
        flags=re.DOTALL | re.IGNORECASE
    )
    if m:
        project_block = m.group(2)
        m_name = re.search(r'name:\s*"([^"]+)"', project_block)
        m_ver = re.search(r'version:\s*"([^"]+)"', project_block)
        if m_name:
            project_name = m_name.group(1).strip()
        if m_ver:
            project_version = m_ver.group(1).strip()

    return {
        "project_name": project_name,
        "project_version": project_version,
    }


def get_schemachange_entry_from_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """
    Récupère la section 'schemachange' du manifest, si présente.
    """
    return manifest.get("schemachange", {}) or {}


def get_dbml_entry_from_manifest(manifest: Dict[str, Any],
                                 project_name: str = "framework_qdd") -> Dict[str, Any]:
    """
    Récupère l'entrée dbml[name=project_name] dans manifest.yml.
    """
    all_dbml = manifest.get("dbml", []) or []
    for entry in all_dbml:
        if entry.get("name") == project_name:
            return entry
    return {}

# ---------------------------------------------------------------------------
# OUTILS DBML : TABLES / COLONNES / FK
# ---------------------------------------------------------------------------

def is_target_table(table, target_name: str) -> bool:
    """
    Vérifie si la table appartient au projet cible (ex: FRAMEWORK_QDD).
    """
    schema = getattr(table, "schema", None)
    full_name = getattr(table, "full_name", None)
    target_name = target_name.upper()

    # Vérification par nom complet (ex: FRAMEWORK_QDD.CUSTOMER)
    if full_name and full_name.upper().startswith(f"{target_name}."):
        return True

    # Vérification par schéma explicite
    if schema and schema.upper() == target_name:
        return True

    return False


def table_qualified_name(table) -> str:
    """
    Retourne le nom qualifié SCHEMA.TABLE en majuscules.
    """
    schema = getattr(table, "schema", None)
    name = getattr(table, "name", "")

    if schema:
        return f"{schema.upper()}.{name.upper()}"
    return name.upper()


def is_pk_column(column) -> bool:
    """
    Détermine si une colonne est PK (en se basant sur column.pk de pydbml).
    """
    return bool(getattr(column, "pk", False))


def is_not_null_column(column) -> bool:
    """
    Détermine si une colonne est NOT NULL (column.not_null).
    """
    return bool(getattr(column, "not_null", False))


def get_fk_targets_for_column(dbml: PyDBML, table, column) -> List[Dict[str, Any]]:
    """
    Retourne la liste des colonnes référencées pour une colonne donnée (si elle participe à une FK).
    """
    fk_list = []

    refs = getattr(dbml, "refs", []) or []
    col_name = column.name

    for ref in refs:
        t1 = ref.table1
        t2 = ref.table2
        cols1 = ref.col1
        cols2 = ref.col2

        if t1 is table and any(c.name == col_name for c in cols1):
            target_table = t2
            target_cols = cols2
        elif t2 is table and any(c.name == col_name for c in cols2):
            target_table = t1
            target_cols = cols1
        else:
            continue

        if not target_table:
            continue

        if target_cols:
            for ref_col in target_cols:
                fk_list.append({
                    "ref_table": target_table,
                    "ref_col": ref_col
                })
        else:
            fk_list.append({
                "ref_table": target_table,
                "ref_col": None
            })

    return fk_list


def fk_targets(dbml: PyDBML, table, column):
    """
    Version de compatibilité : renvoie une liste de (ref_table_obj, ref_col_obj) comme avant.
    """
    results = []
    for item in get_fk_targets_for_column(dbml, table, column):
        results.append((item["ref_table"], item["ref_col"]))
    return results

# ---------------------------------------------------------------------------
# DBML VERSION
# ---------------------------------------------------------------------------

def ensure_dbml_version(conn,
                        project_name: str,
                        project_version: str,
                        dbml_entry: Dict[str, Any]) -> int:
    """
    Vérifie l'existence d'une ligne T_DBML_VERSION correspondant à :
      - DBV_PROJET_NAME
      - DBV_PROJET_VERSION
      - DBV_SCHEMA_CIBLE (si défini dans manifest dbml/schema)
    Si elle n'existe pas, l'insère.
    Retourne DBV_IDF.
    """
    repo_url = dbml_entry.get("url", "") if dbml_entry else ""
    repo_tag = dbml_entry.get("tag", "") if dbml_entry else ""
    schema_cible = dbml_entry.get("schema", "") if dbml_entry else ""

    select_sql = """
        SELECT DBV_IDF
        FROM T_DBML_VERSION
        WHERE DBV_PROJET_NAME = %s
          AND DBV_PROJET_VERSION = %s
          AND COALESCE(DBV_SCHEMA_CIBLE, '') = COALESCE(%s, '')
        ORDER BY DBV_DATE_CREATION DESC
        LIMIT 1
    """

    insert_sql = """
        INSERT INTO T_DBML_VERSION (
            DBV_PROJET_NAME,
            DBV_PROJET_VERSION,
            DBV_REPO_URL,
            DBV_REPO_TAG,
            DBV_SCHEMA_CIBLE,
            DBV_DATE_CREATION
        )
        VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP())
    """

    cur = conn.cursor()
    try:
        cur.execute(select_sql, (project_name, project_version, schema_cible))
        row = cur.fetchone()
        if row:
            dbv_id = row[0]
            logger.info(f"DBML_VERSION déjà existant : DBV_IDF={dbv_id}")
            return dbv_id

        cur.execute(
            insert_sql,
            (project_name, project_version, repo_url, repo_tag, schema_cible),
        )
        conn.commit()

        cur.execute(select_sql, (project_name, project_version, schema_cible))
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Impossible de retrouver la ligne T_DBML_VERSION après insertion.")
        dbv_id = row[0]
        logger.info(f"Nouvelle DBML_VERSION créée : DBV_IDF={dbv_id}")
        return dbv_id
    finally:
        cur.close()

# ---------------------------------------------------------------------------
# CHARGEMENT DES MÉTRIQUES (T_METRIQUE)
# ---------------------------------------------------------------------------

def load_metrics(conn) -> List[Dict[str, Any]]:
    """
    Charge T_METRIQUE, et typage simple :
      - 'completude'
      - 'unicite'
      - 'integrite'
      - 'autre'
    """
    metrics = []
    cur = conn.cursor()
    try:
        sql = """
            SELECT
                MET_IDF,
                MET_NOM_METRIQUE,
                MET_CRITERE_QDD,
                MET_DESCRIPTION
            FROM T_METRIQUE
        """
        cur.execute(sql)
        rows = cur.fetchall()
        for row in rows:
            met_id = row[0]
            met_nom = row[1] or ""
            met_crit = row[2] or ""
            met_desc = row[3]

            nom_lower = met_nom.strip().lower()
            crit_upper = met_crit.strip().upper()

            if "compl" in nom_lower or crit_upper == "EXH":
                type_metr = "completude"
            elif "unic" in nom_lower or crit_upper in ("UNI", "UNQ"):
                type_metr = "unicite"
            elif "intégr" in nom_lower or "integr" in nom_lower or crit_upper in ("INT", "FK", "REF"):
                type_metr = "integrite"
            else:
                type_metr = "autre"

            metrics.append(
                {
                    "id": met_id,
                    "nom": met_nom,
                    "crit": met_crit,
                    "description": met_desc,
                    "type_metr": type_metr,
                }
            )

        logger.info(f"{len(metrics)} métriques chargées.")
        return metrics
    finally:
        cur.close()


def get_metric_by_type(metrics: List[Dict[str, Any]], type_metr: str) -> Optional[Dict[str, Any]]:
    """
    Récupère la première métrique correspondant au type indiqué.
    """
    for m in metrics:
        if m["type_metr"] == type_metr:
            return m
    return None

# ---------------------------------------------------------------------------
# GÉNÉRATION DES TESTCASES
# ---------------------------------------------------------------------------

def format_test_name(critere: str, schema: str, table: str, champ: str, tst_id: int) -> str:
    """
    Construit un nom de test standard :
      CRITERE-SCHEMA-TABLE-CHAMP-TSTID-0
    """
    crit = critere.strip().upper()
    schema = schema.strip().upper()
    table = table.strip().upper()
    champ = champ.strip().upper().replace(",", "_")
    base = f"{crit}-{schema}-{table}-{champ}-{tst_id}-0"
    if len(base) > 255:
        base = base[:255]
    return base


def generate_testcases_from_dbml(
    dbml: PyDBML,
    metrics: List[Dict[str, Any]],
    dbml_version_id: int,
    project_name: str,
    source_cible_id: int = 1
) -> List[Dict[str, Any]]:
    """
    Génère les T_TESTCASE pour toutes les tables cibles trouvées dans le DBML :
      - Complétude (NOT NULL ou PK)
      - Unicité (PK)
      - Intégrité (FK -> table.ref)
    """
    metric_completude = get_metric_by_type(metrics, "completude")
    metric_unicite = get_metric_by_type(metrics, "unicite")
    metric_integrite = get_metric_by_type(metrics, "integrite")

    if not metric_completude:
        logger.warning("Aucune métrique de complétude trouvée.")
    if not metric_unicite:
        logger.warning("Aucune métrique d'unicité trouvée.")
    if not metric_integrite:
        logger.warning("Aucune métrique d'intégrité trouvée.")

    valid_de, valid_jusqua = compute_default_dates()

    testcases: List[Dict[str, Any]] = []
    current_id = 1

    for table in dbml.tables:
        # On utilise ici la nouvelle fonction de filtrage dynamique
        if not is_target_table(table, project_name):
            continue

        table_name = table_qualified_name(table)
        schema_name = getattr(table, "schema", "") or ""
        schema_name = schema_name.upper()
        logger.info(f"Table target détectée : {table_name}")

        columns = getattr(table, "columns", []) or []
        pk_columns = [col for col in columns if is_pk_column(col)]
        not_null_columns = [col for col in columns if is_not_null_column(col)]

        pk_set = {col.name for col in pk_columns}
        nn_set = {col.name for col in not_null_columns}
        all_nn_for_completude = sorted(pk_set.union(nn_set))

        # -------------------------------------------------------------------
        # Complétude
        # -------------------------------------------------------------------
        if metric_completude:
            for col_name in all_nn_for_completude:
                tst_id = current_id
                current_id += 1

                tst_nom = format_test_name(
                    metric_completude["crit"],
                    schema_name,
                    table.name,
                    col_name,
                    tst_id,
                )
                desc = f"Test de complétude - Table {table_name}, Colonne {col_name}"

                testcases.append(
                    {
                        "TST_IDF": tst_id,
                        "TST_NOM_TEST": tst_nom,
                        "TST_SOURCE_CIBLE_ID": source_cible_id,
                        "TST_TABLE_CIBLE": table_name,
                        "TST_CHAMP_CIBLE": col_name,
                        "TST_ID_METRIQUE": metric_completude["id"],
                        "TST_TYPE": "AUTO",
                        "TST_DESCRIPTION": desc,
                        "TST_POIDS": DEFAULT_POIDS,
                        "TST_SEUIL_BORNE_INFERIEURE": DEFAULT_SEUIL_INF,
                        "TST_SEUIL_BORNE_SUPERIEURE": DEFAULT_SEUIL_SUP,
                        "TST_FREQ": DEFAULT_FREQ,
                        "TST_DATE_MISE_A_JOUR": None,
                        "TST_DATE_CREATION": date.today(),
                        "TST_VALIDE_DE": valid_de,
                        "TST_VALIDE_JUSQUA": valid_jusqua,
                        "TST_DBML_VERSION_ID": dbml_version_id,
                        "TST_KIND": "completude",
                    }
                )

        # -------------------------------------------------------------------
        # Unicité (PK)
        # -------------------------------------------------------------------
        if metric_unicite and pk_columns:
            champs = ",".join(col.name for col in pk_columns)
            tst_id = current_id
            current_id += 1

            tst_nom = format_test_name(
                metric_unicite["crit"],
                schema_name,
                table.name,
                champs,
                tst_id,
            )
            desc = f"Test d'unicité - Table {table_name}, Colonnes PK ({champs})"

            testcases.append(
                {
                    "TST_IDF": tst_id,
                    "TST_NOM_TEST": tst_nom,
                    "TST_SOURCE_CIBLE_ID": source_cible_id,
                    "TST_TABLE_CIBLE": table_name,
                    "TST_CHAMP_CIBLE": champs,
                    "TST_ID_METRIQUE": metric_unicite["id"],
                    "TST_TYPE": "AUTO",
                    "TST_DESCRIPTION": desc,
                    "TST_POIDS": DEFAULT_POIDS,
                    "TST_SEUIL_BORNE_INFERIEURE": DEFAULT_SEUIL_INF,
                    "TST_SEUIL_BORNE_SUPERIEURE": DEFAULT_SEUIL_SUP,
                    "TST_FREQ": DEFAULT_FREQ,
                    "TST_DATE_MISE_A_JOUR": None,
                    "TST_DATE_CREATION": date.today(),
                    "TST_VALIDE_DE": valid_de,
                    "TST_VALIDE_JUSQUA": valid_jusqua,
                    "TST_DBML_VERSION_ID": dbml_version_id,
                    "TST_KIND": "unicite",
                }
            )

        # -------------------------------------------------------------------
        # Intégrité (FK)
        # -------------------------------------------------------------------
        if metric_integrite:
            for col in columns:
                fk_list = fk_targets(dbml, table, col)
                if not fk_list:
                    continue

                col_name = col.name

                for fk in fk_list:
                    ref_table = fk[0]
                    ref_col = fk[1]

                    ref_table_name = table_qualified_name(ref_table)
                    ref_col_name = ref_col.name if ref_col is not None else "UNKNOWN"

                    tst_id = current_id
                    current_id += 1

                    tst_nom = format_test_name(
                        metric_integrite["crit"],
                        schema_name,
                        table.name,
                        col_name,
                        tst_id,
                    )
                    desc = (
                        f"Test d'intégrité - Table {table_name}, Colonne {col_name} "
                        f"sur {ref_table_name}.{ref_col_name} de référence"
                    )

                    testcases.append(
                        {
                            "TST_IDF": tst_id,
                            "TST_NOM_TEST": tst_nom,
                            "TST_SOURCE_CIBLE_ID": source_cible_id,
                            "TST_TABLE_CIBLE": table_name,
                            "TST_CHAMP_CIBLE": col_name,
                            "TST_ID_METRIQUE": metric_integrite["id"],
                            "TST_TYPE": "AUTO",
                            "TST_DESCRIPTION": desc,
                            "TST_POIDS": DEFAULT_POIDS,
                            "TST_SEUIL_BORNE_INFERIEURE": DEFAULT_SEUIL_INF,
                            "TST_SEUIL_BORNE_SUPERIEURE": DEFAULT_SEUIL_SUP,
                            "TST_FREQ": DEFAULT_FREQ,
                            "TST_DATE_MISE_A_JOUR": None,
                            "TST_DATE_CREATION": date.today(),
                            "TST_VALIDE_DE": valid_de,
                            "TST_VALIDE_JUSQUA": valid_jusqua,
                            "TST_DBML_VERSION_ID": dbml_version_id,
                            "TST_KIND": "integrite",
                        }
                    )

    logger.info(f"{len(testcases)} testcases générés.")
    return testcases

# ---------------------------------------------------------------------------
# SQL INSERT CONDITIONNEL
# ---------------------------------------------------------------------------

def sql_literal(value: Any) -> str:
    """
    Transforme une valeur Python en littéral SQL.
    """
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (datetime, date)):
        return f"'{value.isoformat()}'"
    s = str(value).replace("'", "''")
    return f"'{s}'"


def build_insert_sql(testcases: List[Dict[str, Any]]) -> str:
    """
    Construit le script SQL :
      INSERT INTO T_TESTCASE ( ... )
      SELECT ...
      WHERE NOT EXISTS ( ... )
    pour chaque testcase.

    On aligne strictement :
      - la liste des colonnes de l'INSERT
      - la liste des valeurs du SELECT
    pour éviter les erreurs "expected N but got M".
    """
    if not testcases:
        return "-- Aucun testcase généré.\n"

    lines = [
        "-- Fichier généré automatiquement par 1_generate_testcases.py",
        f"-- Date de génération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for tc in testcases:
        # Condition d'unicité : on évite les doublons exacts
        cond = (
            "TST_TABLE_CIBLE = {table} "
            "AND TST_CHAMP_CIBLE = {champ} "
            "AND TST_ID_METRIQUE = {met} "
            "AND COALESCE(TST_DESCRIPTION, '') = COALESCE({desc}, '')"
        ).format(
            table=sql_literal(tc["TST_TABLE_CIBLE"]),
            champ=sql_literal(tc["TST_CHAMP_CIBLE"]),
            met=sql_literal(tc["TST_ID_METRIQUE"]),
            desc=sql_literal(tc["TST_DESCRIPTION"]),
        )

        insert = f"""
INSERT INTO T_TESTCASE (
    TST_IDF,
    TST_NOM_TEST,
    TST_SOURCE_CIBLE_ID,
    TST_TABLE_CIBLE,
    TST_CHAMP_CIBLE,
    TST_ID_METRIQUE,
    TST_TYPE,
    TST_DESCRIPTION,
    TST_POIDS,
    TST_SEUIL_BORNE_INFERIEURE,
    TST_SEUIL_BORNE_SUPERIEURE,
    TST_FREQ,
    TST_DATE_MISE_A_JOUR,
    TST_DATE_CREATION,
    TST_VALIDE_DE,
    TST_VALIDE_JUSQUA,
    TST_DBML_VERSION_ID
)
SELECT
    {sql_literal(tc["TST_IDF"])},
    {sql_literal(tc["TST_NOM_TEST"])},
    {sql_literal(tc["TST_SOURCE_CIBLE_ID"])},
    {sql_literal(tc["TST_TABLE_CIBLE"])},
    {sql_literal(tc["TST_CHAMP_CIBLE"])},
    {sql_literal(tc["TST_ID_METRIQUE"])},
    {sql_literal(tc["TST_TYPE"])},
    {sql_literal(tc["TST_DESCRIPTION"])},
    {sql_literal(tc["TST_POIDS"])},
    {sql_literal(tc["TST_SEUIL_BORNE_INFERIEURE"])},
    {sql_literal(tc["TST_SEUIL_BORNE_SUPERIEURE"])},
    {sql_literal(tc["TST_FREQ"])},
    {sql_literal(tc["TST_DATE_MISE_A_JOUR"])},
    {sql_literal(tc["TST_DATE_CREATION"])},
    {sql_literal(tc["TST_VALIDE_DE"])},
    {sql_literal(tc["TST_VALIDE_JUSQUA"])},
    {sql_literal(tc["TST_DBML_VERSION_ID"])}
WHERE NOT EXISTS (
    SELECT 1
    FROM T_TESTCASE
    WHERE {cond}
);
""".strip("\n")

        lines.append(insert)

    return "\n\n".join(lines) + "\n"


def write_sql_file(sql_text: str, target_dir: str = SQL_TARGET_DIR) -> Path:
    """
    Écrit le script SQL dans target/rows-t_testcase-YYYYMMDD-HHMMSS.sql
    """
    p = Path(target_dir)
    p.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"rows-t_testcase-{ts}.sql"
    full_path = p / filename

    full_path.write_text(sql_text, encoding="utf-8")
    logger.info(f"Fichier SQL généré : {full_path}")
    return full_path

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    logger.info("=== Début 1_generate_testcases ===")

    # Chargement manifest + DBML + meta projet
    manifest = load_manifest(MANIFEST_PATH)
    dbml_project_meta = extract_project_meta_from_dbml_file(DBML_PATH)
    dbml = load_dbml(DBML_PATH)

    project_name = dbml_project_meta.get("project_name") or manifest.get("project", {}).get("name", "framework_qdd")
    project_version = dbml_project_meta.get("project_version") or manifest.get("project", {}).get("version", "0.0.0")

    logger.info(f"Projet: {project_name}, version {project_version}")

    conn = get_connection()

####### TEMP CODE #######
    cur = conn.cursor()
    cur.execute("SHOW TABLES")
    rows = cur.fetchall()
    cols = [c[0].lower() for c in cur.description]   # récupère les noms de colonnes
    import pandas as pd
    df = pd.DataFrame(rows, columns=cols)[["name", "schema_name"]]
    print(df)
####### END TEMP CODE #######


    try:
        # DBML_VERSION
        dbml_entry = get_dbml_entry_from_manifest(manifest, project_name=project_name)
        dbml_version_id = ensure_dbml_version(conn, project_name, project_version, dbml_entry)

        # Métriques
        metrics = load_metrics(conn)

        # Génération des testcases
        source_cible_id_int = int(DEFAULT_SOURCE_CIBLE_ID)
        testcases = generate_testcases_from_dbml(
            dbml=dbml,
            metrics=metrics,
            dbml_version_id=dbml_version_id,
            project_name=project_name,
            source_cible_id=source_cible_id_int,
        )

        # Script SQL
        sql_text = build_insert_sql(testcases)
        sql_path = write_sql_file(sql_text, SQL_TARGET_DIR)

        logger.info(f"Génération des testcases terminée. Script : {sql_path}")
    finally:
        conn.close()
        logger.info("Connexion Snowflake fermée.")
    logger.info("=== Fin 1_generate_testcases ===")


if __name__ == "__main__":
    main()
