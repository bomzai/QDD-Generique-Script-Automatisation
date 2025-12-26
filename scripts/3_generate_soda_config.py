#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3_generate_soda_config.py

Objectif :
- Partir d'un template YAML Soda (dev-config-template.yml)
- Remplacer les placeholders par les valeurs des variables d'environnement :
    ${PRIVATE_KEY_PATH}    -> PRIVATE_KEY_PATH  (sans guillemets dans le rendu final)
    ${SNOWFLAKE_ACCOUNT}   -> SNOWFLAKE_ACCOUNT (+ suffixe .privatelink si absent)
    ${SNOWFLAKE_USER}      -> SNOWFLAKE_USER
    ${SNOWFLAKE_WAREHOUSE} -> SNOWFLAKE_WAREHOUSE
    ${SNOWFLAKE_DATABASE}  -> SNOWFLAKE_DATABASE
    ${SNOWFLAKE_ROLE}      -> SNOWFLAKE_ROLE
- Écrire la config finale dans soda/configs/dev-config.yml
"""

from pathlib import Path

from qdd_utils import (
    get_logger,
    env,
    SODA_CONFIG_TEMPLATE,
    SODA_CONFIG_OUTPUT,
)

logger = get_logger("étape 3 : générer la config SODA")

DEFAULT_TEMPLATE = SODA_CONFIG_TEMPLATE
DEFAULT_OUTPUT = SODA_CONFIG_OUTPUT

# Mapping placeholder -> nom de variable d'environnement
PLACEHOLDER_ENV_MAP = {
    "${PRIVATE_KEY_PATH}": "PRIVATE_KEY_PATH",
    "${SNOWFLAKE_ACCOUNT}": "SNOWFLAKE_ACCOUNT",
    "${SNOWFLAKE_USER}": "SNOWFLAKE_USER",
    "${SNOWFLAKE_WAREHOUSE}": "SNOWFLAKE_WAREHOUSE",
    "${SNOWFLAKE_DATABASE}": "SNOWFLAKE_DATABASE",
    "${SNOWFLAKE_ROLE}": "SNOWFLAKE_ROLE",
}


def generate_config(
    template_path: Path = DEFAULT_TEMPLATE,
    output_path: Path = DEFAULT_OUTPUT,
) -> None:
    if not template_path.exists():
        raise FileNotFoundError(f"Template Soda config introuvable : {template_path}")

    text = template_path.read_text(encoding="utf-8")

    # Remplacement de tous les placeholders connus par les valeurs d'env
    for placeholder, env_name in PLACEHOLDER_ENV_MAP.items():
        value = env(env_name, required=True).strip()

        if env_name == "PRIVATE_KEY_PATH":
            # on veut supprimer les guillemets autour du placeholder :
            #   private_key_path: "${PRIVATE_KEY_PATH}"
            # -> private_key_path: /mon/chemin/clé.pem
            quoted_placeholder = f'"{placeholder}"'
            if quoted_placeholder in text:
                text = text.replace(quoted_placeholder, value)
            else:
                text = text.replace(placeholder, value)
        else:
            # Pour les autres, on laisse les guillemets du template faire le job :
            # " ${SNOWFLAKE_USER}" -> "valeur"
            text = text.replace(placeholder, value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")

    logger.info("Fichier Soda config généré : %s", output_path)


if __name__ == "__main__":
    generate_config()