"""
cleanup_centers.py

Merges duplicate HDCenter nodes in Neo4j into their canonical forms.
For each duplicate, all incoming and outgoing relationships are redirected
to the canonical node before the duplicate is deleted.

Usage:
  python scripts/cleanup_centers.py

Requires Neo4j running at bolt://localhost:7687.
"""

from neo4j import GraphDatabase

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# canonical name → list of duplicate names to merge into it
CENTER_MERGE_MAP = {
    "Sacral Center": ["Sacral"],
    "Throat Center": ["Throat"],
    "Solar Plexus Center": [
        "Solar Plexus",
        "Emotional Solar Plexus",
        "Emotional Solar Plexus Center",
        "Emotional Center",
    ],
    "Heart Center": ["Heart", "Ego Center", "Will Center"],
    "Ajna Center": ["Ajna"],
    "Spleen Center": ["Spleen", "Splenic Center"],
    "G Center": ["Self Center"],
    "Head Center": [],
    "Root Center": [],
    "Crown Center": [],
}

# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def merge_duplicate(tx, canonical: str, duplicate: str) -> int:
    """
    Redirect every relationship that touches `duplicate` onto `canonical`,
    then delete `duplicate`. Returns the number of relationships redirected.
    """
    redirected = 0

    # -- Incoming edges: (other) -[r]-> (duplicate)  →  (other) -[r]-> (canonical) --
    incoming = tx.run(
        """
        MATCH (dup {name: $dup})<-[r]-(other)
        RETURN other.name AS other_name, labels(other)[0] AS other_label,
               type(r) AS rel_type, properties(r) AS rel_props
        """,
        dup=duplicate,
    ).data()

    for row in incoming:
        tx.run(
            """
            MATCH (other {name: $other_name})
            MATCH (canon {name: $canon})
            MERGE (other)-[nr:%(rel_type)s]->(canon)
            SET nr += $props
            """ % {"rel_type": row["rel_type"]},
            other_name=row["other_name"],
            canon=canonical,
            props=row["rel_props"],
        )
        redirected += 1

    # -- Outgoing edges: (duplicate) -[r]-> (other)  →  (canonical) -[r]-> (other) --
    outgoing = tx.run(
        """
        MATCH (dup {name: $dup})-[r]->(other)
        RETURN other.name AS other_name, labels(other)[0] AS other_label,
               type(r) AS rel_type, properties(r) AS rel_props
        """,
        dup=duplicate,
    ).data()

    for row in outgoing:
        tx.run(
            """
            MATCH (canon {name: $canon})
            MATCH (other {name: $other_name})
            MERGE (canon)-[nr:%(rel_type)s]->(other)
            SET nr += $props
            """ % {"rel_type": row["rel_type"]},
            canon=canonical,
            other_name=row["other_name"],
            props=row["rel_props"],
        )
        redirected += 1

    # -- Delete the duplicate --
    tx.run("MATCH (dup {name: $dup}) DETACH DELETE dup", dup=duplicate)

    return redirected


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    total_merged = 0
    total_redirected = 0

    with driver.session() as session:
        for canonical, duplicates in CENTER_MERGE_MAP.items():
            if not duplicates:
                continue

            # Ensure the canonical node exists before redirecting into it.
            session.run(
                "MERGE (n:HDCenter {name: $name})",
                name=canonical,
            )

            for duplicate in duplicates:
                # Check if the duplicate actually exists — skip silently if not.
                exists = session.run(
                    "MATCH (n {name: $name}) RETURN count(n) AS c",
                    name=duplicate,
                ).single()["c"]

                if not exists:
                    continue

                redirected = session.execute_write(
                    merge_duplicate, canonical, duplicate
                )
                print(f"[MERGED] {duplicate!r} → {canonical!r}  ({redirected} relationship(s) redirected)")
                total_merged += 1
                total_redirected += redirected

    driver.close()

    print()
    print(f"Done. {total_merged} duplicate node(s) merged, {total_redirected} relationship(s) redirected.")


if __name__ == "__main__":
    main()
