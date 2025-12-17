#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
0_fetch_produit_dbml.py

Objectif :
- Lire manifest.yml pour trouver l'entrée dbml[name=Produit]
- Télécharger le fichier DBML correspondant depuis GitLab
- L'enregistrer sous target/produit.dbml
"""

import urllib.parse
from pathlib import Path
import sys
import requests

from qdd_utils import get_logger, load_manifest, read_text_file

TARGET_DIR = "soda/dbml"
API_KEY_FILE = "target/api-key"

logger = get_logger("fetch_produit_dbml")


def download_dbml(file_path: str, tag: str, token: str, target_dir: str) -> None:
    """
    Télécharge le fichier DBML depuis GitLab.
    Le repo GitLab est pris depuis manifest.yml (url) ; si non défini,
    on garde la valeur par défaut.
    """
    # Repo GitLab par défaut (tu peux le surcharger via manifest.yml)
    default_project = "datahub/dbt/ref/referentiels_data_model"

    manifest = load_manifest()
    dbml_list = manifest.get("dbml", []) or []
    produit_entry = next((e for e in dbml_list if e.get("name") == "Produit"), None)
    repo_url = (produit_entry or {}).get("url", "")
    if repo_url and repo_url.startswith("https://gitlab.mnh.fr/"):
        project = urllib.parse.quote_plus(repo_url.replace("https://gitlab.mnh.fr/", "").rstrip(".git"))
    else:
        project = urllib.parse.quote_plus(default_project)

    file_encoded = urllib.parse.quote_plus(file_path)
    url = (
        f"https://gitlab.mnh.fr/api/v4/projects/{project}/repository/files/"
        f"{file_encoded}/raw?ref={tag}"
    )

    logger.info("Téléchargement du DBML depuis : %s", url)
    resp = requests.get(url, headers={"PRIVATE-TOKEN": token})

    if resp.status_code != 200:
        logger.error("Erreur %s lors du téléchargement depuis %s", resp.status_code, url)
        logger.error("Contenu de la réponse (200 premiers caractères) : %s", resp.text[:200])
        sys.exit(1)

    Path(target_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(target_dir) / "produit.dbml"
    out_path.write_bytes(resp.content)
    logger.info("DBML Produit téléchargé avec succès : %s", out_path)


def main() -> None:
    manifest = load_manifest()
    token = read_text_file(API_KEY_FILE)

    dbml_list = manifest.get("dbml", [])
    if not isinstance(dbml_list, list):
        logger.error("Le champ 'dbml' doit être une liste dans manifest.yml")
        sys.exit(1)

    produit = next((item for item in dbml_list if item.get("name") == "Produit"), None)
    if not produit:
        logger.error("Aucune entrée 'Produit' trouvée dans la section 'dbml' du manifest.yml")
        sys.exit(1)

    tag = produit.get("tag")
    if not tag:
        logger.error("Champ 'tag' manquant pour 'Produit' dans le manifest.yml")
        sys.exit(1)

    file_path = produit.get("file_path") or "dbml/produit/core.dbml"
    logger.info("Téléchargement du DBML Produit @ tag=%s, file_path=%s", tag, file_path)
    download_dbml(file_path, tag, token, TARGET_DIR)


if __name__ == "__main__":
    main()
