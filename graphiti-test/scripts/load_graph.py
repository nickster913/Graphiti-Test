"""
load_graph.py
Loads output/hd_graph_seed.json into Neo4j via direct Cypher.
No Graphiti. No frameworks. Just the neo4j driver.

Run:
    pip install neo4j
    python scripts/load_graph.py
"""

import json
from pathlib import Path

from neo4j import GraphDatabase

# ── Config ────────────────────────────────────────────────────────────────────
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"
JSON_FILE = Path(__file__).parent.parent / "output" / "hd_graph_seed.json"

# ── Node type → label mapping ─────────────────────────────────────────────────
NODE_LABELS = {
    "HDType": "HDType",
    "HDCenter": "HDCenter",
    "HDGate": "HDGate",
    "HDChannel": "HDChannel",
    "HDProfile": "HDProfile",
    "HDAuthority": "HDAuthority",
    "HDConcept": "HDConcept",
    "RayJaiTeaching": "RayJaiTeaching",
    "Person": "Person",
}


def load_nodes(tx, nodes):
    for node in nodes:
        node_type = node.get("type")
        label = NODE_LABELS.get(node_type, "Entity")
        name = node.get("name")

        if not name:
            print(f"  [SKIP] Node with no name: {node}")
            continue

        # Build properties dict — everything except 'type'
        props = {k: v for k, v in node.items() if k != "type" and v is not None}

        query = f"""
        MERGE (n:{label} {{name: $name}})
        SET n += $props
        """
        tx.run(query, name=name, props=props)
        print(f"  [NODE] ({label}) {name}")


def load_edges(tx, edges, node_name_to_label):
    for edge in edges:
        from_name = edge.get("from")
        to_name = edge.get("to")
        rel_type = edge.get("type")

        if not from_name or not to_name or not rel_type:
            print(f"  [SKIP] Incomplete edge: {edge}")
            continue

        # Look up labels for MATCH specificity
        from_label = node_name_to_label.get(from_name, "")
        to_label = node_name_to_label.get(to_name, "")

        from_match = f":{from_label}" if from_label else ""
        to_match = f":{to_label}" if to_label else ""

        query = f"""
        MATCH (a{from_match} {{name: $from_name}})
        MATCH (b{to_match} {{name: $to_name}})
        MERGE (a)-[r:{rel_type}]->(b)
        RETURN a.name, type(r), b.name
        """
        result = tx.run(query, from_name=from_name, to_name=to_name)
        record = result.single()
        if record:
            print(f"  [EDGE] {from_name} -[{rel_type}]-> {to_name}")
        else:
            print(f"  [WARN] Could not create edge: {from_name} -[{rel_type}]-> {to_name} (nodes not found?)")


def main():
    # Load JSON
    with open(JSON_FILE, encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    print(f"Loaded {len(nodes)} nodes and {len(edges)} edges from {JSON_FILE}\n")

    # Build name → label lookup for edge matching
    node_name_to_label = {}
    for node in nodes:
        name = node.get("name")
        label = NODE_LABELS.get(node.get("type"), "Entity")
        if name:
            node_name_to_label[name] = label

    # Connect and load
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    with driver.session() as session:
        print("── Loading nodes ─────────────────────────────────")
        session.execute_write(load_nodes, nodes)

        print("\n── Loading edges ─────────────────────────────────")
        session.execute_write(load_edges, edges, node_name_to_label)

    driver.close()

    print(f"\n✓ Done. {len(nodes)} nodes, {len(edges)} edges written to Neo4j.")
    print("  Open http://localhost:7474 and run:")
    print("  MATCH (n)-[r]->(m) RETURN n, r, m")


if __name__ == "__main__":
    main()