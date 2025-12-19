#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
2_generate_soda_checks.py

Objectif :
- Lire T_TESTCASE + T_METRIQUE sur Snowflake
- Déduire le type de test : completude / unicite / integrite / autre
- Générer des fichiers SodaCL (YAML) par table dans target/soda_checks/
  au format Soda 2.0, par exemple :

  checks for SCH_REF_PRODUIT.T_PRODUIT:
    - duplicate_percent(PRD_CODE_PRODUIT) between 0.0 and 0.0:
        name: "COH-PRODUIT-T_PRODUIT-PRD_CODE_PRODUIT-1-0"
        attributes:
          description: "Test d'unicité ..."
          tags: "QDD,TECH"
    - missing_percent(PRD_LIB_PRODUIT) between 0.0 and 0.0:
        name: "EXH-PRODUIT-T_PRODUIT-PRD_LIB_PRODUIT-7-0"
        attributes:
          description: "Test de complétude ..."
          tags: "QDD,TECH"
    - invalid_count-int-PRODUIT.T_PRODUIT-PRD_APERITEUR_IDF between 0.0 and 0.0:
        name: "COH-PRODUIT-T_PRODUIT-PRD_APERITEUR_IDF-14-0"
        invalid_count-int-PRODUIT.T_PRODUIT-PRD_APERITEUR_IDF query: |
          SELECT COUNT(*) AS "invalid_count-int-PRODUIT.T_PRODUIT-PRD_APERITEUR_IDF"
          FROM SCH_REF_PRODUIT.T_PRODUIT t
          WHERE t.PRD_APERITEUR_IDF IS NOT NULL
            AND t.PRD_APERITEUR_IDF NOT IN (
              SELECT PER_IDF
              FROM SCH_REF_PERSONNE.T_ORGANISATION
            )
        attributes:
          description: "Test d'intégrité ..."
          tags: "QDD,TECH"
"""

from pathlib import Path
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Optional

from qdd_utils import get_logger, connect_snowflake_from_env

logger = get_logger("étape 2 : génerer les checks SODA")

OUTPUT_DIR = Path("target/soda_checks")


# ---------------------------------------------------------------------------
# AIDE : type de métrique
# ---------------------------------------------------------------------------

def classify_metric(metric_name: str, critere: str) -> str:
    """
    Classe la métrique en :
    - 'completude'
    - 'unicite'
    - 'integrite'
    - 'autre'
    en fonction du nom de métrique et du critère QDD.
    """
    nom_lower = (metric_name or "").strip().lower()
    crit_upper = (critere or "").strip().upper()

    if "compl" in nom_lower or crit_upper == "EXH":
        return "completude"
    if "unic" in nom_lower or crit_upper in ("UNI", "UNQ"):
        return "unicite"
    if "intégr" in nom_lower or "integr" in nom_lower or crit_upper in ("INT", "FK", "REF"):
        return "integrite"
    return "autre"


def parse_integrity_ref(description: str) -> Optional[Tuple[str, str]]:
    """
    Cherche le pattern dans TST_DESCRIPTION, par ex. :

      '... sur PERSONNE.T_ORGANISATION.PER_IDF de référence'
      '... sur LOV.T_T_PROD.CODE de référence'
      '... sur PRODUIT.T_PRODUIT.PRD_CODE_PRODUIT de réference'

    Retourne (table_logique, colonne) = ('PERSONNE.T_ORGANISATION', 'PER_IDF')
    ou None si non trouvé.
    """
    if not description:
        return None
    desc = description.upper()
    marker = " SUR "
    marker2 = " DE RÉFÉRENCE"
    marker2_alt = " DE REFERENCE"

    try:
        if marker not in desc:
            return None
        idx = desc.index(marker) + len(marker)

        if marker2 in desc:
            end_idx = desc.index(marker2, idx)
        elif marker2_alt in desc:
            end_idx = desc.index(marker2_alt, idx)
        else:
            return None

        segment = desc[idx:end_idx].strip()
        # segment = 'PERSONNE.T_ORGANISATION.PER_IDF'
        parts = segment.split(".")
        if len(parts) == 3:
            schema = parts[0].strip()
            table = parts[1].strip()
            col = parts[2].strip()
            table_full = f"{schema}.{table}"
            return table_full, col
        return None
    except Exception:
        return None


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
        return f"SCH_REF_{table_cible.upper()}"
    schema, table = table_cible.split(".", 1)
    schema_up = schema.strip().upper()
    table_up = table.strip().upper()
    if schema_up.startswith("SCH_REF_"):
        return f"{schema_up}.{table_up}"
    return f"SCH_REF_{schema_up}.{table_up}"


def ref_dataset_name_from_table(ref_table: str) -> str:
    """
    Transforme une table logique 'SCHEMA.T_TABLE' en table physique 'SCH_REF_SCHEMA.T_TABLE'.

    Exemple :
      'PERSONNE.T_ORGANISATION' -> 'SCH_REF_PERSONNE.T_ORGANISATION'
      'LOV.T_T_PROD'            -> 'SCH_REF_LOV.T_T_PROD'
    """
    if not ref_table or "." not in ref_table:
        return ref_table or ""

    schema, table = ref_table.split(".", 1)
    schema_up = schema.strip().upper()
    table_up = table.strip().upper()

    if schema_up.startswith("SCH_REF_"):
        return f"{schema_up}.{table_up}"
    return f"SCH_REF_{schema_up}.{table_up}"


# ---------------------------------------------------------------------------
# RÉCUPÉRATION DES TESTCASES
# ---------------------------------------------------------------------------

def load_testcases_with_metrics(conn) -> List[Dict[str, Any]]:
    """
    Joint T_TESTCASE et T_METRIQUE, retourne une liste de dicts.
    """
    sql = """
        SELECT
            T.TST_IDF,
            T.TST_NOM_TEST,
            T.TST_TABLE_CIBLE,
            T.TST_CHAMP_CIBLE,
            T.TST_SEUIL_BORNE_INFERIEURE,
            T.TST_SEUIL_BORNE_SUPERIEURE,
            T.TST_DESCRIPTION,
            M.MET_NOM_METRIQUE,
            M.MET_CRITERE_QDD
        FROM T_TESTCASE T
        JOIN T_METRIQUE M
          ON T.TST_ID_METRIQUE = M.MET_IDF
    """
    cur = conn.cursor()
    try:
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [c[0] for c in cur.description]

        results: List[Dict[str, Any]] = []
        for r in rows:
            row_dict = dict(zip(cols, r))
            results.append(row_dict)

        logger.info("%d testcases chargés pour génération Soda.", len(results))
        return results
    finally:
        cur.close()


# ---------------------------------------------------------------------------
# GÉNÉRATION YAML (format Soda 2.0)
# ---------------------------------------------------------------------------

def generate_soda_checks_for_table(dataset: str, testcases: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Construit le dictionnaire Python représentant le YAML SodaCL pour un dataset,
    au format :

      checks for <dataset>:
        - missing_percent(COL) between x and y:
            name: "..."
            attributes:
              description: "..."
              tags: "QDD,TECH"

        - duplicate_percent(COL1, COL2) between x and y:
            name: "..."
            attributes:
              description: "..."
              tags: "QDD,TECH"

        - invalid_count-int-PRODUIT.T_PRODUIT-COL between x and y:
            name: "..."
            invalid_count-int-PRODUIT.T_PRODUIT-COL query: |
              SELECT COUNT(*) AS "invalid_count-int-PRODUIT.T_PRODUIT-COL"
              FROM SCH_REF_PRODUIT.T_PRODUIT t
              WHERE t.COL IS NOT NULL
                AND t.COL NOT IN (
                  SELECT REF_COL
                  FROM SCH_REF_SCHEMA.REF_TABLE
                )
            attributes:
              description: "..."
              tags: "QDD,TECH"
    """
    checks: List[Dict[str, Any]] = []

    for tc in testcases:
        met_name = tc["MET_NOM_METRIQUE"]
        critere = tc["MET_CRITERE_QDD"]
        kind = classify_metric(met_name, critere)

        col = (tc["TST_CHAMP_CIBLE"] or "").strip()
        table_cible = (tc["TST_TABLE_CIBLE"] or "").strip()
        desc = tc.get("TST_DESCRIPTION") or ""
        seuil_inf = tc.get("TST_SEUIL_BORNE_INFERIEURE")
        seuil_sup = tc.get("TST_SEUIL_BORNE_SUPERIEURE")

        # Valeurs par défaut si None -> 0.0
        low = float(seuil_inf) if seuil_inf is not None else 0.0
        high = float(seuil_sup) if seuil_sup is not None else 0.0

        attributes = {
            "description": desc,
            "tags": "QDD,TECH",
        }

        # ---------------------------------------------------------
        # COMPLETUDE -> missing_percent(COL) between x and y:
        # ---------------------------------------------------------
        if kind == "completude":
            if not col:
                logger.warning("Testcase %s sans colonne cible pour complétude, ignoré.", tc["TST_IDF"])
                continue

            header = f"missing_percent({col}) between {low} and {high}"
            body = {
                "name": tc["TST_NOM_TEST"],
                "attributes": attributes,
            }
            checks.append({header: body})
            continue

        # ---------------------------------------------------------
        # UNICITE -> duplicate_percent(COL1, COL2) between x and y:
        # ---------------------------------------------------------
        if kind == "unicite":
            if not col:
                logger.warning("Testcase %s sans colonne cible pour unicité, ignoré.", tc["TST_IDF"])
                continue

            cols = [c.strip() for c in col.split(",") if c.strip()]
            if not cols:
                logger.warning("Testcase %s avec liste de colonnes vide pour unicité, ignoré.", tc["TST_IDF"])
                continue

            if len(cols) == 1:
                col_expr = cols[0]
            else:
                col_expr = ", ".join(cols)

            header = f"duplicate_percent({col_expr}) between {low} and {high}"
            body = {
                "name": tc["TST_NOM_TEST"],
                "attributes": attributes,
            }
            checks.append({header: body})
            continue

        # ---------------------------------------------------------
        # INTEGRITE -> invalid_count-int-PRODUIT.T_PRODUIT-COL between x and y:
        #             + <metric_name> query: |
        # ---------------------------------------------------------
        if kind == "integrite":
            ref_info = parse_integrity_ref(desc)
            if not ref_info:
                logger.warning(
                    "Impossible d'extraire la table de référence pour le testcase %s (desc tronquée: '%s')",
                    tc["TST_IDF"],
                    desc[:200],
                )
                continue

            ref_table_logical, ref_col = ref_info
            ref_table_physical = ref_dataset_name_from_table(ref_table_logical)

            # metric_name : invalid_count-int-PRODUIT.T_PRODUIT-PRD_COL
            metric_name = f"invalid_count-int-{table_cible}-{col}"

            header = f"{metric_name} between {low} and {high}"

            query_sql = (
                f'SELECT COUNT(*) AS "{metric_name}"\n'
                f"FROM {dataset} t\n"
                f"WHERE t.{col} IS NOT NULL\n"
                f"  AND t.{col} NOT IN (\n"
                f"    SELECT {ref_col}\n"
                f"    FROM {ref_table_physical}\n"
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
        # AUTRES METRIQUES : ignorées pour l'instant
        # ---------------------------------------------------------
        logger.info("Testcase %s (metrique '%s') classé en 'autre', ignoré.", tc["TST_IDF"], met_name)

    # Clé racine au format "checks for <dataset>"
    return {f"checks for {dataset}": checks}


def write_soda_yaml(dataset: str, yaml_dict: Dict[str, Any]):
    """
    Écrit le YAML SodaCL dans target/soda_checks/<dataset>.yml
    Format :

      checks for <dataset>:
        - <check expression> between ... and ...:
            name: "..."
            ...
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filename = dataset.replace(".", "_") + ".yml"
    path = OUTPUT_DIR / filename

    def dump_value(v, indent: int = 0) -> List[str]:
        sp = " " * indent
        lines: List[str] = []

        if isinstance(v, dict):
            for k, val in v.items():
                # Chaîne multi-ligne -> block scalar |
                if isinstance(val, str) and "\n" in val:
                    lines.append(f"{sp}{k}: |")
                    for line in val.splitlines():
                        lines.append(f"{sp}  {line}")
                elif isinstance(val, (dict, list)):
                    lines.append(f"{sp}{k}:")
                    lines.extend(dump_value(val, indent + 2))
                else:
                    if isinstance(val, str):
                        lines.append(f'{sp}{k}: "{val}"')
                    else:
                        lines.append(f"{sp}{k}: {val}")

        elif isinstance(v, list):
            for item in v:
                # Cas typique : { "duplicate_percent(...) between ..." : { ... } }
                if isinstance(item, dict) and len(item) == 1:
                    (k, val), = item.items()
                    # On met le "-" et la clé sur la même ligne :
                    lines.append(f"{sp}- {k}:")
                    # Et on indente le contenu interne 4 espaces plus loin
                    lines.extend(dump_value(val, indent + 4))
                elif isinstance(item, (dict, list)):
                    # Fallback générique si jamais
                    lines.append(f"{sp}-")
                    lines.extend(dump_value(item, indent + 2))
                else:
                    if isinstance(item, str):
                        lines.append(f'{sp}- "{item}"')
                    else:
                        lines.append(f"{sp}- {item}")

        else:
            if isinstance(v, str):
                lines.append(f'{sp}"{v}"')
            else:
                lines.append(f"{sp}{v}")

        return lines

    lines = dump_value(yaml_dict, indent=0)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Fichier SodaCL généré : %s", path)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    conn = connect_snowflake_from_env(role_env="SNOWFLAKE_ROLE")

    try:
        rows = load_testcases_with_metrics(conn)

        # Groupement par table cible
        by_table: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            table_cible = r["TST_TABLE_CIBLE"]
            by_table[table_cible].append(r)

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
