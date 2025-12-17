#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from qdd_utils import get_logger, connect_snowflake_from_env, load_dbml_object

logger = get_logger("étape 1 : Parsing DBML -> Snowflake")

DBML_FILE = "votre_fichier.dbml" # <--- METTEZ LE NOM DE VOTRE FICHIER ICI

def insert_testcase(cursor, data):
    """Insère un testcase s'il n'existe pas déjà."""
    query = """
    INSERT INTO T_TESTCASE (TST_NOM_TEST, TST_TABLE_CIBLE, TST_CHAMP_CIBLE, TST_ID_METRIQUE, TST_DESCRIPTION, TST_SEUIL_BORNE_INFERIEURE, TST_SEUIL_BORNE_SUPERIEURE)
    SELECT %s, %s, %s, %s, %s, 0, 0
    WHERE NOT EXISTS (SELECT 1 FROM T_TESTCASE WHERE TST_NOM_TEST = %s)
    """
    cursor.execute(query, (data['nom'], data['table'], data['champ'], data['id_metrique'], data['desc'], data['nom']))

def main():
    conn = connect_snowflake_from_env(role_env="SNOWFLAKE_ROLE")
    parsed = load_dbml_object(DBML_FILE)
    cursor = conn.cursor()

    try:
        for table in parsed.tables:
            schema = table.schema or "PUBLIC"
            full_table = f"{schema}.{table.name}".upper()
            
            for col in table.columns:
                # 1. Test d'Unicité (si PK) -> ID Métrique 1
                if col.pk:
                    insert_testcase(cursor, {
                        "nom": f"UNI-{full_table}-{col.name}".upper(),
                        "table": full_table, "champ": col.name, "id_metrique": 1,
                        "desc": f"Unicité automatique sur PK {col.name}"
                    })

                # 2. Test de Complétude (si non null ou PK) -> ID Métrique 2
                if not col.nullable or col.pk:
                    insert_testcase(cursor, {
                        "nom": f"EXH-{full_table}-{col.name}".upper(),
                        "table": full_table, "champ": col.name, "id_metrique": 2,
                        "desc": f"Complétude automatique sur {col.name}"
                    })

            # 3. Test d'Intégrité (si REF détectée) -> ID Métrique 3
            for ref in parsed.refs:
                for rel in ref.col_rels:
                    if rel.table1.name == table.name:
                        t1, c1 = f"{rel.table1.schema}.{rel.table1.name}".upper(), rel.column1.name.upper()
                        t2, c2 = f"{rel.table2.schema}.{rel.table2.name}".upper(), rel.column2.name.upper()
                        insert_testcase(cursor, {
                            "nom": f"INT-{t1}-{c1}".upper(),
                            "table": t1, "champ": c1, "id_metrique": 3,
                            "desc": f"Intégrité sur {t2}.{c2} de référence"
                        })

        conn.commit()
        logger.info("Parsing DBML terminé et T_TESTCASE mis à jour.")
    finally:
        conn.close()

if __name__ == "__main__":
    main()