#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qdd_utils.py

Fonctions utilitaires partagées pour les scripts QDD / Produit :

- Gestion de Logs
- Chargement des variables d'environnement
- Chargement et lecture du manifest YAML
- Lecture de fichiers texte
- Chargement de la clé privée Snowflake
- Connexion à Snowflake depuis les variables d'environnement
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
import snowflake.connector
import yaml
from cryptography.hazmat.primitives import serialization

# Dossier racine du projet (là où se trouve manifest.yml, soda/, target/, etc.)
BASE_DIR = Path("./")

# manifest.yml
MANIFEST_PATH = str((BASE_DIR / "manifest.yml").resolve())

# Dossier target/
TARGET_DIR = Path(os.getenv("QDD_TARGET_DIR", "target"))

# Dossier DBML (par défaut soda/dbml/)
DBML_DIR = Path(os.getenv("QDD_DBML_DIR", BASE_DIR / "dbml"))

# Fichier token GitLab (écrit par run.sh)
GITLAB_TOKEN_FILE = Path(os.getenv("GITLAB_TOKEN_FILE", TARGET_DIR / "api-key"))

# Dossier des fichiers de checks Soda
SODA_CHECKS_DIR = Path(os.getenv("QDD_SODA_CHECKS_DIR", TARGET_DIR / "soda_checks"))

# Template et fichier final de config Soda
SODA_CONFIG_TEMPLATE = Path("soda/configs/dev-config-template.yml")
SODA_CONFIG_OUTPUT = Path("soda/configs/dev-config.yml")

# Fichier results.json de Soda
RESULTS_JSON_PATH = Path(
    os.getenv("QDD_RESULTS_JSON", TARGET_DIR / "results.json")
)

# (Optionnel) Dossier où tu génères les SQL de testcases
SQL_TARGET_DIR = Path(os.getenv("SQL_TARGET_DIR", TARGET_DIR))

# --- SQL safe helpers --------------------------------------------------------
import re

_IDENT_RE = re.compile(r"^[A-Z0-9_]+$", re.IGNORECASE)

def q_ident(name: str) -> str:
    name = (name or "").strip()
    if not _IDENT_RE.match(name):
        raise ValueError(f"Identifiant non autorisé: {name!r}")
    return f'"{name.upper()}"'

def q_qualified(obj: str) -> str:
    parts = [p.strip() for p in (obj or "").split(".") if p.strip()]
    if not parts:
        raise ValueError(f"Objet qualifié vide: {obj!r}")
    if len(parts) > 3:
        raise ValueError(f"Objet trop qualifié (max 3 niveaux): {obj!r}")
    return ".".join(q_ident(p) for p in parts)

def q_str(v: str) -> str:
    v = "" if v is None else str(v)
    return "'" + v.replace("'", "''") + "'"

# --- YAML writer -------------------------------------------------------------

def write_yaml_file(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# GESTION DE LOGS
# ---------------------------------------------------------------------------

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Retourne un logger configuré.
    Niveau contrôlé par QDD_LOG_LEVEL (DEBUG, INFO, WARNING, ERROR).
    """
    logger_name = name or "qdd"
    logger = logging.getLogger(logger_name)

    # Configure le root logger une seule fois
    if not logging.getLogger().handlers:
        level_name = os.getenv("QDD_LOG_LEVEL", "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

    return logger


# ---------------------------------------------------------------------------
# CHARGEMENT DES VARIABLES D'ENVIRONNEMENT
# ---------------------------------------------------------------------------

def env(name: str, default: Any = None, required: bool = False) -> Any:
    """
    Récupère une variable d'environnement.

    :param name: Nom de la variable d'environnement.
    :param default: Valeur par défaut.
    :param required: Si True, lève RuntimeError si la variable est absente ou vide.
    """
    value = os.environ.get(name, default)
    if required and (value is None or str(value).strip() == ""):
        logger = get_logger(__name__)
        logger.error("Variable d'environnement manquante : %s", name)
        raise RuntimeError(f"Missing required env var: {name}")
    return value


# ---------------------------------------------------------------------------
# PARSING MANIFEST & FICHIERS TEXTE
# ---------------------------------------------------------------------------

def load_manifest(manifest_path: str) -> Dict[str, Any]:
    """
    Charge le fichier manifest.yml et renvoie un dict.
    """
    logger = get_logger("qdd_utils")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.yml introuvable : {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    logger.info(f"manifest.yml chargé depuis {manifest_path}")
    return data

def classify_metric(metric_name: str, critere: str) -> str:
    """
    Classe la métrique en :
    - 'completude'
    - 'unicite'
    - 'integrite'
    - 'tracabilite'
    - 'autre'
    en fonction du nom de la métrique et du critère QDD.

    :param metric_name: MET_NOM_METRIQUE
    :param critere: MET_CRITERE_QDD (ex : EXH, COH, TRA, ...)
    """
    nom_lower = (metric_name or "").strip().lower()
    crit_upper = (critere or "").strip().upper()

    # Complétude (EXH)
    if "compl" in nom_lower or crit_upper == "EXH":
        return "completude"

    # Unicité
    if "unic" in nom_lower or crit_upper in ("UNI", "UNQ"):
        return "unicite"

    # Traçabilité (description de colonnes)
    # - critère QDD = 'TRA'
    if "traç" in nom_lower or crit_upper == "TRA":
        return "tracabilite"

    # Intégrité de référentiel / FK
    if "intégr" in nom_lower or "integr" in nom_lower or crit_upper in ("INT", "FK", "REF"):
        return "integrite"

    return "autre"

def read_text_file(path: str, strip: bool = True) -> str:
    """
    Lit le contenu d'un fichier texte et le retourne.

    :param path: Chemin du fichier.
    :param strip: Si True, applique .strip().
    """
    logger = get_logger(__name__)
    file_path = Path(path)

    if not file_path.exists():
        logger.error("Fichier introuvable : %s", file_path)
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    content = file_path.read_text(encoding="utf-8")
    if strip:
        content = content.strip()

    logger.info("Fichier texte lu : %s", file_path)
    return content


# ---------------------------------------------------------------------------
# CHARGEMENT DE LA CLÉ PRIVÉE SNOWFLAKE
# ---------------------------------------------------------------------------

def load_private_key(private_key_path: str, passphrase: Optional[str] = None):
    """
    Charge la clé privée au format PEM pour Snowflake.

    :param private_key_path: Chemin du fichier PEM.
    :param passphrase: Passphrase éventuelle (None si non chiffrée).
    :return: Objet clé privée à passer à snowflake.connector.connect().
    """
    logger = get_logger(__name__)

    if not private_key_path:
        raise RuntimeError("PRIVATE_KEY_PATH non défini.")

    pem_path = Path(private_key_path)
    if not pem_path.exists():
        logger.error("Clé privée introuvable : %s", pem_path)
        raise FileNotFoundError(f"Clé privée introuvable : {pem_path}")

    with pem_path.open("rb") as f:
        key_data = f.read()

    password_bytes = passphrase.encode("utf-8") if passphrase else None
    private_key = serialization.load_pem_private_key(
        key_data,
        password=password_bytes,
    )
    logger.info("Clé privée chargée depuis %s", pem_path)
    return private_key


# ---------------------------------------------------------------------------
# CONNEXION SNOWFLAKE
# ---------------------------------------------------------------------------

def connect_snowflake_from_env(
    private_key_env: str = "PRIVATE_KEY_PATH",
    user_env: str = "SNOWFLAKE_USER",
    account_env: str = "SNOWFLAKE_ACCOUNT",
    warehouse_env: str = "SNOWFLAKE_WAREHOUSE",
    database_env: str = "SNOWFLAKE_DATABASE",
    schema_env: str = "SNOWFLAKE_SCHEMA",
    role_env: Optional[str] = None,
) -> "snowflake.connector.SnowflakeConnection":
    """
    Crée une connexion Snowflake en lisant les paramètres via les variables d'env.
    """
    logger = get_logger(__name__)

    pk_path = env(private_key_env, required=True)
    private_key = load_private_key(pk_path, passphrase=os.getenv("PRIVATE_KEY_PASSPHRASE"))
    snowflake_account = env(account_env, required=True)

    #if ".privatelink" not in snowflake_account:
    #    snowflake_account = snowflake_account + ".privatelink"

    connect_args = {
        "user": env(user_env, required=True),
        "account": snowflake_account,
        "warehouse": env(warehouse_env, required=True),
        "database": env(database_env, required=True),
        "schema": env(schema_env, required=True),
        "private_key": private_key,
    }

    if role_env:
        role_value = env(role_env, default=None, required=False)
        if role_value:
            connect_args["role"] = role_value

    try:
        conn = snowflake.connector.connect(**connect_args)
        logger.info(
            "Connexion Snowflake OK (user=%s, account=%s, warehouse=%s, database=%s, schema=%s)",
            connect_args.get("user"),
            connect_args.get("account"),
            connect_args.get("warehouse"),
            connect_args.get("database"),
            connect_args.get("schema"),
        )
        return conn
    except Exception as e:
        logger.error("Erreur de connexion à Snowflake : %s", e)
        raise


# Nom logique du DBML (par défaut : premier dbml[].name du manifest, sinon "Produit")
def get_default_dbml_name(manifest_path: str) -> str:
    try:
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            entries = data.get("dbml", []) or []
            for entry in entries:
                name = entry.get("name")
                if name:
                    return name
    except Exception:
        logger.warning(
            "Impossible de déduire DBML_NAME depuis manifest.yml, utilisation de 'Produit'."
        )
    return "Produit"

def get_target_dbml_schema(manifest_path: str, dbml_name: str) -> str:
    """
    Lit le manifest et renvoie le schéma de référence du domaine DBML :
    - d'abord dbml[].logical_schema si présent
    - sinon dbml[].schema (ex: SCH_REF_PRODUIT)
    - si ça commence par SCH_REF_, on enlève le préfixe pour obtenir le schéma logique (PRODUIT)
    - sinon on renvoie la valeur telle quelle.

    Si rien n'est trouvé, renvoie "".
    """
    try:
        if not os.path.exists(manifest_path):
            return ""
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        entries = data.get("dbml", []) or []
        for entry in entries:
            if entry.get("name") == dbml_name:
                # Permet d'ajouter un champ dédié logique si tu veux
                candidate = (
                    entry.get("logical_schema")
                    or entry.get("schema")
                    or ""
                )
                candidate = (candidate or "").strip()
                if not candidate:
                    return ""
                candidate_up = candidate.upper()
                if candidate_up.startswith("SCH_REF_"):
                    # SCH_REF_PRODUIT => PRODUIT
                    candidate_up = candidate_up[len("SCH_REF_"):]
                return candidate_up
    except Exception as e:
        logger.warning(
            "Impossible de déduire le schéma logique DBML depuis manifest.yml : %s",
            e,
        )
        return ""

    return ""



# ---------------------------------------------------------------------------
# DBML VERSION (factorisé)
# ---------------------------------------------------------------------------

def ensure_dbml_version(
    conn,
    project_name: str,
    project_version: str,
    dbml_entry: Dict[str, Any],
    logger: Optional[logging.Logger] = None,
) -> int:
    """    Vérifie l'existence d'une ligne T_DBML_VERSION correspondant à :
      - DBV_PROJET_NAME
      - DBV_PROJET_VERSION
      - DBV_SCHEMA_CIBLE (si défini dans manifest dbml/schema)

    Si elle n'existe pas, l'insère.
    Retourne DBV_IDF.

    Note : factorisée ici pour être utilisée par 1_generate_testcases.py et 2_generate_soda_checks.py.
    """
    logger = logger or get_logger("qdd_utils")

    repo_url = (dbml_entry or {}).get("url", "")
    repo_tag = (dbml_entry or {}).get("tag", "")
    schema_cible = (dbml_entry or {}).get("schema", "")

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
            dbv_id = int(row[0])
            logger.info("DBML_VERSION déjà existant : DBV_IDF=%s", dbv_id)
            return dbv_id

        cur.execute(insert_sql, (project_name, project_version, repo_url, repo_tag, schema_cible))
        conn.commit()

        cur.execute(select_sql, (project_name, project_version, schema_cible))
        row = cur.fetchone()
        if not row:
            raise RuntimeError("Impossible de retrouver la ligne T_DBML_VERSION après insertion.")
        dbv_id = int(row[0])
        logger.info("Nouvelle DBML_VERSION créée : DBV_IDF=%s", dbv_id)
        return dbv_id
    finally:
        cur.close()
DBML_NAME = os.getenv("customer", get_default_dbml_name(str(MANIFEST_PATH)))
# Chemin du fichier DBML principal
DBML_PATH = DBML_DIR / f"{DBML_NAME.lower()}.dbml"