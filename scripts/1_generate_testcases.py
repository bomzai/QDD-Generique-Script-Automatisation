#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import hashlib
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Any, List, Optional
from qdd_utils import load_manifest, get_target_dbml_schema
from qdd_utils import (
    get_logger,
    connect_snowflake_from_env,
    classify_metric,
    get_target_dbml_schema,
    MANIFEST_PATH,
    DBML_NAME,
    DBML_PATH,
    ensure_dbml_version,
    TABLE_TESTCASE,
    TABLE_METRIQUE,
)

from pydbml import PyDBML

logger = get_logger("étape 1 : génerer les testcases")

# ---------------------------------------------------------------------------
# PARAMÈTRES QDD
# ---------------------------------------------------------------------------
# Génération des tests
# ➜ Désormais initialisés en dur, puis surchargés par manifest.yml
DEFAULT_SOURCE_CIBLE_ID = "1"
DEFAULT_POIDS = 1.0
DEFAULT_SEUIL_INF = 0.0
DEFAULT_SEUIL_SUP = 0.0
DEFAULT_FREQ = "J"

# Validité
DEFAULT_VALIDE_DE = os.getenv("QDD_DEFAULT_VALIDE_DE", "1900-01-01")
DEFAULT_VALIDE_JUSQUA = os.getenv("QDD_DEFAULT_VALIDE_JUSQUA", "2099-12-31")

# ---------------------------------------------------------------------------
# UTILITAIRES DATES
# ---------------------------------------------------------------------------

def parse_date(s: str) -> date:
    """
    Convertit une chaîne 'YYYY-MM-DD' en date.
    """
    return datetime.strptime(s, "%Y-%m-%d").date()


def compute_default_dates() -> (date, date): # type: ignore
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

def apply_testcase_defaults_from_manifest(manifest: Dict[str, Any]) -> None:
    """
    Applique les valeurs par défaut des T_TESTCASE à partir du manifest.yml.

    Section attendue :

        defaults_testcases_values:
          tst_poids: 1
          tst_freq: "J"
          tst_seuil_borne_inferieur: 0
          tst_seuil_borne_superieur: 0
          # optionnel :
          # tst_source_cible_id: 1

    Si une clé est absente, on conserve la valeur
    déjà présente dans les constantes globales.
    """
    global DEFAULT_SOURCE_CIBLE_ID, DEFAULT_POIDS, DEFAULT_SEUIL_INF, DEFAULT_SEUIL_SUP, DEFAULT_FREQ

    defaults = manifest.get("defaults_testcases_values") or {}
    if not isinstance(defaults, dict) or not defaults:
        logger.warning(
            "Section 'defaults_testcases_values' absente ou vide dans manifest.yml, "
            "utilisation des valeurs codées en dur dans le script."
        )
        return

    # Source cible (optionnelle)
    if "tst_source_cible_id" in defaults and defaults["tst_source_cible_id"] is not None:
        try:
            DEFAULT_SOURCE_CIBLE_ID = str(int(defaults["tst_source_cible_id"]))
        except (TypeError, ValueError):
            logger.warning(
                "Impossible de convertir defaults_testcases_values.tst_source_cible_id en int (%r), "
                "valeur inchangée.",
                defaults["tst_source_cible_id"],
            )

    # Poids
    if "tst_poids" in defaults and defaults["tst_poids"] is not None:
        try:
            DEFAULT_POIDS = float(defaults["tst_poids"])
        except (TypeError, ValueError):
            logger.warning(
                "Impossible de convertir defaults_testcases_values.tst_poids en float (%r), "
                "valeur inchangée.",
                defaults["tst_poids"],
            )

    # Seuils
    if "tst_seuil_borne_inferieur" in defaults and defaults["tst_seuil_borne_inferieur"] is not None:
        try:
            DEFAULT_SEUIL_INF = float(defaults["tst_seuil_borne_inferieur"])
        except (TypeError, ValueError):
            logger.warning(
                "Impossible de convertir defaults_testcases_values.tst_seuil_borne_inferieur en float (%r), "
                "valeur inchangée.",
                defaults["tst_seuil_borne_inferieur"],
            )

    if "tst_seuil_borne_superieur" in defaults and defaults["tst_seuil_borne_superieur"] is not None:
        try:
            DEFAULT_SEUIL_SUP = float(defaults["tst_seuil_borne_superieur"])
        except (TypeError, ValueError):
            logger.warning(
                "Impossible de convertir defaults_testcases_values.tst_seuil_borne_superieur en float (%r), "
                "valeur inchangée.",
                defaults["tst_seuil_borne_superieur"],
            )

    # Fréquence
    if "tst_freq" in defaults and defaults["tst_freq"] is not None:
        DEFAULT_FREQ = str(defaults["tst_freq"])
        if not DEFAULT_FREQ:
            DEFAULT_FREQ = "J"
            logger.warning(
                "defaults_testcases_values.tst_freq est vide, utilisation de 'J' par défaut."
            )


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


def get_dbml_entry_from_manifest(manifest: Dict[str, Any],
                                 project_name: str = "customer") -> Dict[str, Any]:
    """
    Récupère l'entrée dbml[name=project_name] dans manifest.yml.
    """
    all_dbml = manifest.get("dbml", []) or []
    for entry in all_dbml:
        if entry.get("name") == project_name:
            return entry
    return {}

def load_business_validations_from_manifest(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Charge les règles de validation métier depuis manifest.yml.

    Returns:
        Liste des configurations de validation avec structure:
        {
            'validation_id': 'positive_balance',
            'validation_name': 'Solde Compte Positif',
            'targets': [{'schema': 'X', 'table': 'Y', 'column': 'Z', 'rule_sql': '...'}]
        }
    """
    validations = manifest.get("business_validations", []) or []

    if not validations:
        logger.info("Aucune validation métier définie dans manifest.yml")
        return []

    logger.info(f"{len(validations)} validation(s) métier chargée(s) depuis manifest")
    return validations

# ---------------------------------------------------------------------------
# OUTILS DBML : TABLES / COLONNES / FK
# ---------------------------------------------------------------------------

def is_ref_table(table) -> bool:
    """
    Détermine si une table appartient au schéma logique ciblé.
    Adapté pour CUSTOMER_SCHEMA.
    """
    TARGET_DBML_SCHEMA = get_target_dbml_schema(MANIFEST_PATH, DBML_NAME)
    
    # On définit les schémas autorisés (le vôtre + le défaut)
    allowed_targets = ["CUSTOMER_SCHEMA"]
    if TARGET_DBML_SCHEMA:
        allowed_targets.append(TARGET_DBML_SCHEMA.upper())

    schema = (getattr(table, "schema", "") or "").upper()
    full_name = (getattr(table, "full_name", "") or "").upper()
    name = (getattr(table, "name", "") or "").upper()

    # Vérification si l'un des schémas autorisés est présent
    for target in allowed_targets:
        if schema == target:
            return True
        if full_name.startswith(target + "."):
            return True
        if name.startswith(target + "."):
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
# GÉNÉRATION D'ID DÉTERMINISTE (TST_IDF)
# ---------------------------------------------------------------------------

def _norm(s: Any) -> str:
    return (str(s) if s is not None else "").strip().upper()

def _norm_table(schema: str, table: str) -> str:
    t = _norm(table)
    if "." in t:
        return t
    s = _norm(schema)
    return f"{s}.{t}" if s else t

def build_testcase_key(
    *,
    kind: str,
    schema_name: str,
    table_name: str,
    champ_key: str,
    metric_id: Any,
    source_cible_id: Any,
    dbml_version_id: Any,
    ref_target: Optional[str] = None,
) -> str:
    """
    Clé fonctionnelle stable du testcase.
    - champ_key : version normalisée/triée si besoin (ex: PK composite)
    - ref_target : utilisé pour distinguer les tests d'intégrité multiples sur une même colonne
    """
    parts = [
        _norm(kind),
        _norm_table(schema_name, table_name),
        _norm(champ_key).replace(" ", ""),
        str(metric_id),
        str(source_cible_id),
        str(dbml_version_id),
    ]
    if ref_target:
        parts.append(_norm(ref_target))
    return "|".join(parts)

def build_tst_idf(key: str) -> str:
    """
    ID stable, court, < 50 chars.
    """
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"TC_{h[:32]}"

# ---------------------------------------------------------------------------
# CHARGEMENT DES MÉTRIQUES (T_METRIQUE)
# ---------------------------------------------------------------------------

def load_metrics(conn) -> List[Dict[str, Any]]:
    """
    Charge T_METRIQUE, et typage simple :
      - 'completude'
      - 'unicite'
      - 'integrite'
      - 'tracabilite'
      - 'autre'
    """
    metrics = []
    cur = conn.cursor()
    try:
        sql = f"""
            SELECT
                MET_IDF,
                MET_NOM_METRIQUE,
                MET_TYPE,
                MET_DESCRIPTION
            FROM {TABLE_METRIQUE}
        """
        cur.execute(sql)
        rows = cur.fetchall()
        for row in rows:
            met_id = row[0]
            met_nom = row[1] or ""
            met_crit = row[2] or ""
            met_desc = row[3]

            type_metr = classify_metric(met_nom, met_crit)

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

def get_or_create_validation_metric(
    conn,
    validation_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Récupère ou crée dynamiquement une métrique de validation.
    Permet des métriques pilotées par manifest sans INSERT manuel.
    """
    validation_id = validation_config.get("validation_id", "")
    validation_name = validation_config.get("validation_name", "")
    description = validation_config.get("description", "")
    formula = validation_config.get("formula", "")

    # ID de métrique stable
    met_idf = f"MET_VAL_{validation_id.upper()}"

    cur = conn.cursor()
    try:
        # Vérifier si la métrique existe
        sql_select = f"""
            SELECT MET_IDF, MET_NOM_METRIQUE, MET_TYPE, MET_DESCRIPTION
            FROM {TABLE_METRIQUE}
            WHERE MET_IDF = %s
        """
        cur.execute(sql_select, (met_idf,))
        row = cur.fetchone()

        if row:
            logger.info(f"Utilisation métrique existante: {met_idf}")
            return {
                "id": row[0],
                "nom": row[1] or "",
                "crit": row[2] or "",
                "description": row[3],
                "type_metr": "validation_metier",
            }

        # Créer nouvelle métrique avec formule si disponible
        sql_insert = f"""
            INSERT INTO {TABLE_METRIQUE} (
                MET_IDF, MET_NOM_METRIQUE, MET_TYPE,
                MET_DESCRIPTION, MET_FORMULE_CALCUL, MET_DATE_MISE_A_JOUR
            )
            VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP())
        """

        met_nom = f"Validation Métier - {validation_name}"
        met_type = "VAL_METIER"

        # Utiliser la formule si fournie, sinon NULL
        met_formule = formula if formula else None

        cur.execute(sql_insert, (met_idf, met_nom, met_type, description, met_formule))
        conn.commit()

        logger.info(f"Création métrique: {met_idf} avec formule: {met_formule}")

        return {
            "id": met_idf,
            "nom": met_nom,
            "crit": met_type,
            "description": description,
            "type_metr": "validation_metier",
        }
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

def format_test_name(critere: str, schema: str, table: str, champ: str, tst_idf: str) -> str:
    """
    Construit un nom de test standard :
      CRITERE-SCHEMA-TABLE-CHAMP-<HASH8>-0
    """
    crit = critere.strip().upper()
    schema = schema.strip().upper()
    table = table.strip().upper()
    champ = champ.strip().upper().replace(",", "_")
    suffix = (tst_idf or "").replace("TC_", "")[:8].upper()

    base = f"{crit}-{schema}-{table}-{champ}-{suffix}-0"
    if len(base) > 255:
        base = base[:255]
    return base

def generate_testcases_from_dbml(
    dbml: PyDBML,
    metrics: List[Dict[str, Any]],
    dbml_version_id: int,
    source_cible_id: int = 1,
    manifest: Dict[str, Any] = None,
    conn = None
) -> List[Dict[str, Any]]:
    """
    Génère les T_TESTCASE pour toutes les tables PRODUIT trouvées dans le DBML :
      - Complétude (NOT NULL ou PK)
      - Unicité (PK)
      - Intégrité (FK -> table.ref)
      - Traçabilité (description de colonne) : 1 test par colonne
    """
    metric_completude = get_metric_by_type(metrics, "completude")
    metric_unicite = get_metric_by_type(metrics, "unicite")
    metric_integrite = get_metric_by_type(metrics, "integrite")
    metric_tracabilite = get_metric_by_type(metrics, "tracabilite")

    if not metric_completude:
        logger.warning("Aucune métrique de complétude trouvée.")
    if not metric_unicite:
        logger.warning("Aucune métrique d'unicité trouvée.")
    if not metric_integrite:
        logger.warning("Aucune métrique d'intégrité trouvée.")
    if not metric_tracabilite:
        logger.warning("Aucune métrique de traçabilité trouvée.")

    valid_de, valid_jusqua = compute_default_dates()

    testcases: List[Dict[str, Any]] = []

    # Charger les validations métier une seule fois (en dehors de la boucle)
    business_validations = []
    if manifest:
        business_validations = load_business_validations_from_manifest(manifest)

    for table in dbml.tables:
        if not is_ref_table(table):
            continue

        table_name = table_qualified_name(table)
        schema_name = getattr(table, "schema", "") or ""
        schema_name = schema_name.upper()
        logger.info(f"Table cible détectée : {table_name}")

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
                key = build_testcase_key(
                    kind="completude",
                    schema_name=schema_name,
                    table_name=table.name,
                    champ_key=col_name,
                    metric_id=metric_completude["id"],
                    source_cible_id=source_cible_id,
                    dbml_version_id=dbml_version_id,
                )
                tst_idf = build_tst_idf(key)

                tst_nom = format_test_name(
                    metric_completude["crit"],
                    schema_name,
                    table.name,
                    col_name,
                    tst_idf,
                )
                desc = f"Test de complétude - Table {table_name}, Colonne {col_name}"

                testcases.append(
                    {
                        "TST_IDF": tst_idf,
                        "TST_NOM_TEST": tst_nom,
                        "TST_SOURCE_CIBLE_ID": source_cible_id,
                        "TST_TABLE_CIBLE": table_name,
                        "TST_CHAMP_CIBLE": col_name,
                        "TST_IDF_METRIQUE": metric_completude["id"],
                        "TST_TYPE": "AUTO_GENERER",
                        "TST_DESCRIPTION": desc,
                        "TST_POIDS": DEFAULT_POIDS,
                        "TST_SEUIL_BORNE_INFERIEURE": DEFAULT_SEUIL_INF,
                        "TST_SEUIL_BORNE_SUPERIEURE": DEFAULT_SEUIL_SUP,
                        "TST_FREQ": DEFAULT_FREQ,
                        "TST_DATE_MISE_A_JOUR": None,
                        "TST_DATE_CREATION": date.today(),
                        "TST_VALIDE_DE": valid_de,
                        "TST_VALIDE_JUSQUA": valid_jusqua,
                        "TST_ID_DBML_VERSION": dbml_version_id,
                        "TST_KIND": "completude",
                    }
                )

        # -------------------------------------------------------------------
        # Unicité (PK)
        # -------------------------------------------------------------------
        if metric_unicite and pk_columns:
            # Affichage inchangé pour le champ cible
            champs_display = ",".join(col.name for col in pk_columns)
            # Version triée uniquement pour stabiliser l'ID
            champs_key = ",".join(sorted(col.name for col in pk_columns))

            key = build_testcase_key(
                kind="unicite",
                schema_name=schema_name,
                table_name=table.name,
                champ_key=champs_key,
                metric_id=metric_unicite["id"],
                source_cible_id=source_cible_id,
                dbml_version_id=dbml_version_id,
            )
            tst_idf = build_tst_idf(key)

            tst_nom = format_test_name(
                metric_unicite["crit"],
                schema_name,
                table.name,
                champs_display,
                tst_idf,
            )
            desc = f"Test d'unicité - Table {table_name}, Colonnes PK ({champs_display})"

            testcases.append(
                {
                    "TST_IDF": tst_idf,
                    "TST_NOM_TEST": tst_nom,
                    "TST_SOURCE_CIBLE_ID": source_cible_id,
                    "TST_TABLE_CIBLE": table_name,
                    "TST_CHAMP_CIBLE": champs_display,
                    "TST_IDF_METRIQUE": metric_unicite["id"],
                    "TST_TYPE": "AUTO_GENERER",
                    "TST_DESCRIPTION": desc,
                    "TST_POIDS": DEFAULT_POIDS,
                    "TST_SEUIL_BORNE_INFERIEURE": DEFAULT_SEUIL_INF,
                    "TST_SEUIL_BORNE_SUPERIEURE": DEFAULT_SEUIL_SUP,
                    "TST_FREQ": DEFAULT_FREQ,
                    "TST_DATE_MISE_A_JOUR": None,
                    "TST_DATE_CREATION": date.today(),
                    "TST_VALIDE_DE": valid_de,
                    "TST_VALIDE_JUSQUA": valid_jusqua,
                    "TST_ID_DBML_VERSION": dbml_version_id,
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

                    # ref_target pour éviter collisions quand une même colonne a plusieurs FK
                    ref_target = f"{ref_table_name}.{ref_col_name}"

                    key = build_testcase_key(
                        kind="integrite",
                        schema_name=schema_name,
                        table_name=table.name,
                        champ_key=col_name,
                        metric_id=metric_integrite["id"],
                        source_cible_id=source_cible_id,
                        dbml_version_id=dbml_version_id,
                        ref_target=ref_target,
                    )
                    tst_idf = build_tst_idf(key)

                    tst_nom = format_test_name(
                        metric_integrite["crit"],
                        schema_name,
                        table.name,
                        col_name,
                        tst_idf,
                    )
                    desc = (
                        f"Test d'intégrité - Table {table_name}, Colonne {col_name} "
                        f"sur {ref_table_name}.{ref_col_name} de référence"
                    )

                    testcases.append(
                        {
                            "TST_IDF": tst_idf,
                            "TST_NOM_TEST": tst_nom,
                            "TST_SOURCE_CIBLE_ID": source_cible_id,
                            "TST_TABLE_CIBLE": table_name,
                            "TST_CHAMP_CIBLE": col_name,
                            "TST_IDF_METRIQUE": metric_integrite["id"],
                            "TST_TYPE": "AUTO_GENERER",
                            "TST_DESCRIPTION": desc,
                            "TST_POIDS": DEFAULT_POIDS,
                            "TST_SEUIL_BORNE_INFERIEURE": DEFAULT_SEUIL_INF,
                            "TST_SEUIL_BORNE_SUPERIEURE": DEFAULT_SEUIL_SUP,
                            "TST_FREQ": DEFAULT_FREQ,
                            "TST_DATE_MISE_A_JOUR": None,
                            "TST_DATE_CREATION": date.today(),
                            "TST_VALIDE_DE": valid_de,
                            "TST_VALIDE_JUSQUA": valid_jusqua,
                            "TST_ID_DBML_VERSION": dbml_version_id,
                            "TST_KIND": "integrite",
                        }
                    )

        # -------------------------------------------------------------------
        # Traçabilité (description de colonne) - 1 test par colonne
        # -------------------------------------------------------------------
        if metric_tracabilite:
            for col in columns:
                col_name = col.name

                key = build_testcase_key(
                    kind="tracabilite",
                    schema_name=schema_name,
                    table_name=table.name,
                    champ_key=col_name,
                    metric_id=metric_tracabilite["id"],
                    source_cible_id=source_cible_id,
                    dbml_version_id=dbml_version_id,
                )
                tst_idf = build_tst_idf(key)

                tst_nom = format_test_name(
                    metric_tracabilite["crit"],
                    schema_name,
                    table.name,
                    col_name,
                    tst_idf,
                )

                desc = (
                    f"Test de traçabilité - Table {table_name}, "
                    f"Colonne {col_name} (description renseignée dans Snowflake)"
                )

                testcases.append(
                    {
                        "TST_IDF": tst_idf,
                        "TST_NOM_TEST": tst_nom,
                        "TST_SOURCE_CIBLE_ID": source_cible_id,
                        "TST_TABLE_CIBLE": table_name,
                        "TST_CHAMP_CIBLE": col_name,
                        "TST_IDF_METRIQUE": metric_tracabilite["id"],
                        "TST_TYPE": "AUTO_GENERER",
                        "TST_DESCRIPTION": desc,
                        "TST_POIDS": DEFAULT_POIDS,
                        "TST_SEUIL_BORNE_INFERIEURE": DEFAULT_SEUIL_INF,
                        "TST_SEUIL_BORNE_SUPERIEURE": DEFAULT_SEUIL_SUP,
                        "TST_FREQ": DEFAULT_FREQ,
                        "TST_DATE_MISE_A_JOUR": None,
                        "TST_DATE_CREATION": date.today(),
                        "TST_VALIDE_DE": valid_de,
                        "TST_VALIDE_JUSQUA": valid_jusqua,
                        "TST_ID_DBML_VERSION": dbml_version_id,
                        "TST_KIND": "tracabilite",
                    }
                )
        # -------------------------------------------------------------------
        # VALIDATION MÉTIER (Business Validations)
        # -------------------------------------------------------------------
        if business_validations:
            for validation_config in business_validations:
                targets = validation_config.get("targets", [])

                for target in targets:
                    target_schema = target.get("schema", "").upper()
                    target_table = target.get("table", "").upper()
                    target_column = target.get("column", "").upper()
                    rule_sql = target.get("rule_sql", "")
                    threshold_min = target.get("threshold_min", DEFAULT_SEUIL_INF)
                    threshold_max = target.get("threshold_max", DEFAULT_SEUIL_SUP)

                    # Vérifier si cette cible correspond à la table en cours
                    if schema_name != target_schema or table.name.upper() != target_table:
                        continue

                    # Vérifier que la colonne existe
                    col_exists = any(col.name.upper() == target_column for col in columns)
                    if not col_exists:
                        logger.warning(
                            f"Colonne {target_column} non trouvée dans {table_name}, ignorée"
                        )
                        continue

                    # Récupérer/créer métrique (nécessite conn - voir modification 3.4)
                    validation_metric = get_or_create_validation_metric(conn, validation_config)

                    # Générer clé stable pour testcase
                    validation_id = validation_config.get("validation_id", "")
                    key = build_testcase_key(
                        kind="validation_metier",
                        schema_name=schema_name,
                        table_name=table.name,
                        champ_key=target_column,
                        metric_id=validation_metric["id"],
                        source_cible_id=source_cible_id,
                        dbml_version_id=dbml_version_id,
                        ref_target=validation_id,
                    )
                    tst_idf = build_tst_idf(key)

                    tst_nom = format_test_name(
                        validation_metric["crit"],
                        schema_name,
                        table.name,
                        target_column,
                        tst_idf,
                    )

                    validation_name = validation_config.get("validation_name", "")
                    desc = (
                        f"Test de validation métier - Table {table_name}, "
                        f"Colonne {target_column} - Règle: {validation_name}"
                    )

                    # Stocker rule_sql dans description pour Script 2
                    desc_with_rule = f"{desc} - RULE_SQL: {rule_sql}"

                    testcases.append({
                        "TST_IDF": tst_idf,
                        "TST_NOM_TEST": tst_nom,
                        "TST_SOURCE_CIBLE_ID": source_cible_id,
                        "TST_TABLE_CIBLE": table_name,
                        "TST_CHAMP_CIBLE": target_column,
                        "TST_IDF_METRIQUE": validation_metric["id"],
                        "TST_TYPE": "AUTO_GENERER",
                        "TST_DESCRIPTION": desc_with_rule,
                        "TST_POIDS": DEFAULT_POIDS,
                        "TST_SEUIL_BORNE_INFERIEURE": threshold_min,
                        "TST_SEUIL_BORNE_SUPERIEURE": threshold_max,
                        "TST_FREQ": DEFAULT_FREQ,
                        "TST_DATE_MISE_A_JOUR": None,
                        "TST_DATE_CREATION": date.today(),
                        "TST_VALIDE_DE": valid_de,
                        "TST_VALIDE_JUSQUA": valid_jusqua,
                        "TST_ID_DBML_VERSION": dbml_version_id,
                        "TST_KIND": "validation_metier",
                    })
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
    """
    if not testcases:
        return "-- Aucun testcase généré.\n"

    lines = [
        "-- Fichier généré automatiquement par 1_generate_testcases.py",
        f"-- Date de génération : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    for tc in testcases:
        cond = "TST_IDF = {idf}".format(
            idf=sql_literal(tc["TST_IDF"]),
        )

        insert = f"""
INSERT INTO {TABLE_TESTCASE} (
    TST_IDF,
    TST_NOM_TEST,
    TST_SOURCE_CIBLE_ID,
    TST_TABLE_CIBLE,
    TST_CHAMP_CIBLE,
    TST_IDF_METRIQUE,
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
    TST_ID_DBML_VERSION
)
SELECT
    {sql_literal(tc["TST_IDF"])},
    {sql_literal(tc["TST_NOM_TEST"])},
    {sql_literal(tc["TST_SOURCE_CIBLE_ID"])},
    {sql_literal(tc["TST_TABLE_CIBLE"])},
    {sql_literal(tc["TST_CHAMP_CIBLE"])},
    {sql_literal(tc["TST_IDF_METRIQUE"])},
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
    {sql_literal(tc["TST_ID_DBML_VERSION"])}
WHERE NOT EXISTS (
    SELECT 1
    FROM {TABLE_TESTCASE}
    WHERE {cond}
);
""".strip("\n")

        lines.append(insert)

    return "\n\n".join(lines) + "\n"


def execute_sql_on_snowflake(sql_text: str, conn) -> None:
    """
    Exécute les statements SQL sur Snowflake.
    Extrait tous les blocs INSERT, UPDATE, DELETE, MERGE.
    """
    if not sql_text:
        logger.info("Aucun SQL à exécuter.")
        return
    
    if "-- Aucun testcase généré." in sql_text:
        logger.info("Aucun testcase à synchroniser.")
        return

    logger.info("Début de l'exécution du script SQL sur Snowflake...")
    
    # Extraction de tous les statements DML valides
    pattern = r'((?:INSERT\s+INTO|UPDATE|DELETE\s+FROM|MERGE\s+INTO)\s+[\s\S]*?;)'
    statements = re.findall(pattern, sql_text, re.IGNORECASE)
    
    if not statements:
        logger.warning("Aucun statement DML trouvé dans le SQL généré.")
        return
    
    cur = conn.cursor()
    count = 0
    
    try:
        for stmt in statements:
            cur.execute(stmt)
            count += 1
        logger.info(f"Synchronisation réussie : {count} requêtes exécutées.")
    except Exception as e:
        logger.error(f"Erreur lors de l'exécution Snowflake : {e}")
        raise
    finally:
        cur.close()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    logger.info("=== Début 1_generate_testcases ===")

    # Chargement manifest + DBML + meta projet
    manifest = load_manifest(MANIFEST_PATH)

    # ➜ Application des valeurs par défaut depuis manifest.yml
    apply_testcase_defaults_from_manifest(manifest)

    dbml_project_meta = extract_project_meta_from_dbml_file(DBML_PATH)
    dbml = load_dbml(DBML_PATH)

    project_name = dbml_project_meta.get("project_name") #or manifest.get("project", {}).get("name", "Produit")
    project_version = dbml_project_meta.get("project_version") #or manifest.get("project", {}).get("version", "0.0.0")

    logger.info(f"Projet: {project_name}, version {project_version}")

    conn = get_connection()
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
            source_cible_id=source_cible_id_int,
            manifest=manifest,
            conn=conn       
        )

        # Script SQL
        sql_text = build_insert_sql(testcases)
        #sql_path = write_sql_file(sql_text, SQL_TARGET_DIR)
        execute_sql_on_snowflake(sql_text, conn)
        logger.info(f"Génération des testcases terminée.")
    finally:
        conn.close()
        logger.info("Connexion Snowflake fermée.")
    logger.info("=== Fin 1_generate_testcases ===")


if __name__ == "__main__":
    main()