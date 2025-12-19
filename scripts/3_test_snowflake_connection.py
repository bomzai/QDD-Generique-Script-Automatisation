#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
4_test_snowflake_connection.py

Objectif :
- Tester la connexion à Snowflake en utilisant les variables d'environnement
  utilisées dans la config Soda :

    SNOWFLAKE_ACCOUNT
    SNOWFLAKE_USER
    PRIVATE_KEY_PATH
    SNOWFLAKE_ROLE
    SNOWFLAKE_DATABASE
    SNOWFLAKE_WAREHOUSE

- Vérifier que la connexion fonctionne en exécutant une requête simple :
    SELECT current_account(), current_warehouse(), current_database(),
           current_role(), current_version();
"""

from pathlib import Path

import snowflake.connector
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

from qdd_utils import get_logger, env

logger = get_logger("étape 4 : test connexion Snowflake")


def _load_private_key(private_key_path: str):
    """
    Charge la clé privée au format PKCS8 (non chiffrée) et la convertit
    au format attendu par le connecteur Snowflake.
    """
    key_path = Path(private_key_path)
    if not key_path.exists():
        raise FileNotFoundError(f"Clé privée introuvable : {key_path}")

    logger.info("Chargement de la clé privée depuis : %s", key_path)

    with key_path.open("rb") as f:
        private_key = serialization.load_pem_private_key(
            f.read(),
            password=None,  # None si la clé n'est pas chiffrée
            backend=default_backend(),
        )

    private_key_der = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    return private_key_der


def test_snowflake_connection() -> None:
    # 1) Récupération des variables d'environnement (obligatoires)
    account = env("SNOWFLAKE_ACCOUNT", required=True).strip()
    user = env("SNOWFLAKE_USER", required=True).strip()
    role = env("SNOWFLAKE_ROLE", required=True).strip()
    database = env("SNOWFLAKE_DATABASE", required=True).strip()
    warehouse = env("SNOWFLAKE_WAREHOUSE", required=True).strip()
    private_key_path = env("PRIVATE_KEY_PATH", required=True).strip()

    logger.info("Test de connexion à Snowflake...")
    logger.info("Compte : %s", account)
    logger.info("Utilisateur : %s", user)
    logger.info("Rôle : %s", role)
    logger.info("Database : %s", database)
    logger.info("Warehouse : %s", warehouse)

    # 2) Chargement de la clé privée
    private_key = _load_private_key(private_key_path)

    # 3) Connexion à Snowflake
    try:
        conn = snowflake.connector.connect(
            account=account,
            user=user,
            private_key=private_key,
            warehouse=warehouse,
            database=database,
            role=role,
        )

        logger.info("Connexion établie, exécution de la requête de test...")

        # 4) Requête simple pour vérifier le contexte
        query = """
        SELECT
            current_account()     AS account,
            current_warehouse()   AS warehouse,
            current_database()    AS database,
            current_role()        AS role,
            current_version()     AS version
        """
        cursor = conn.cursor()
        cursor.execute(query)
        result = cursor.fetchone()

        if result:
            account_res, wh_res, db_res, role_res, version_res = result
            logger.info("Connexion OK ✅")
            logger.info("Account   : %s", account_res)
            logger.info("Warehouse : %s", wh_res)
            logger.info("Database  : %s", db_res)
            logger.info("Role      : %s", role_res)
            logger.info("Version   : %s", version_res)
        else:
            logger.warning("La requête de test n'a retourné aucun résultat.")

    except Exception as exc:
        logger.error("Échec de la connexion à Snowflake ❌")
        logger.exception(exc)
        raise
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    test_snowflake_connection()