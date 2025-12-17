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

import snowflake.connector
import yaml
from cryptography.hazmat.primitives import serialization


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

def load_manifest(path: str = "manifest.yml") -> Dict[str, Any]:
    """
    Charge le manifest YAML et retourne un dict.
    """
    logger = get_logger(__name__)
    manifest_path = Path(path)

    if not manifest_path.exists():
        logger.error("Fichier manifest introuvable : %s", manifest_path)
        raise FileNotFoundError(f"manifest.yml introuvable : {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    logger.info("manifest.yml chargé depuis %s", manifest_path)
    return data


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

    if ".privatelink" not in snowflake_account:
        snowflake_account = snowflake_account + ".privatelink"

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
