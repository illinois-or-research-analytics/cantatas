import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Perform Leiden clustering on a nodelist and edgelist.")
    parser.add_argument("--nodelist", required=True, help="Path to the nodelist CSV.")
    parser.add_argument("--edgelist", required=True, help="Path to the edgelist CSV.")
    parser.add_argument("--output", required=True, help="Path to the output CSV.")
    parser.add_argument("--node_col", default="node_id", help="Column name for node IDs in nodelist.")
    parser.add_argument("--source_col", default="source", help="Column name for source nodes in edgelist.")
    parser.add_argument("--target_col", default="target", help="Column name for target nodes in edgelist.")
    parser.add_argument("--directed", action="store_true", help="Treat the graph as directed.")
    parser.add_argument("--resolution", type=float, default=0.01, help="Resolution parameter for Leiden.")
    parser.add_argument("--objective", default="CPM", choices=["CPM", "modularity"], help="Objective function for Leiden.")
    
    args = parser.parse_args()

    print(f"Reading nodelist from {args.nodelist}...")
    nodes_df = pd.read_csv(args.nodelist)
    print(f"Reading edgelist from {args.edgelist}...")
    edges_df = pd.read_csv(args.edgelist)

    # Ensure columns exist
    if args.node_col not in nodes_df.columns:
        raise ValueError(f"Column '{args.node_col}' not found in nodelist.")
    if args.source_col not in edges_df.columns or args.target_col not in edges_df.columns:
        raise ValueError(f"Source/Target columns '{args.source_col}', '{args.target_col}' not found in edgelist.")

    # To handle non-continuous or string node IDs, we'll create a mapping to continuous integers [0, N-1]
    unique_nodes = nodes_df[args.node_col].unique()
    num_nodes = len(unique_nodes)
    print(f"Graph has {num_nodes} unique nodes.")

    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    print("Mapping edges to continuous indices...")
    # Filter edges that are actually in the nodelist
    valid_edges = edges_df[
        edges_df[args.source_col].isin(node_to_idx) & 
        edges_df[args.target_col].isin(node_to_idx)
    ]
    
    num_ignored = len(edges_df) - len(valid_edges)
    if num_ignored > 0:
        print(f"Warning: Ignored {num_ignored} edges with nodes not present in the nodelist.")

    source_indices = valid_edges[args.source_col].map(node_to_idx).tolist()
    target_indices = valid_edges[args.target_col].map(node_to_idx).tolist()
    edge_list = list(zip(source_indices, target_indices))

    print("Building igraph...")
    try:
        import igraph as ig
    except ImportError:
        raise ImportError("The 'igraph' python package is required. Install it using 'pip install python-igraph' or 'conda install -c conda-forge python-igraph'.")

    g = ig.Graph(n=num_nodes, edges=edge_list, directed=args.directed)
    
    print(f"Running Leiden clustering (objective={args.objective}, resolution={args.resolution})...")
    
    # Try using igraph's built-in community_leiden (available in newer versions)
    try:
        partition = g.community_leiden(
            objective_function=args.objective, 
            resolution_parameter=args.resolution
        )
        membership = partition.membership
    except AttributeError:
        print("igraph.Graph.community_leiden not found. Falling back to 'leidenalg' package...")
        try:
            import leidenalg
            if args.objective == "modularity":
                partition_type = leidenalg.ModularityVertexPartition
            else:
                partition_type = leidenalg.CPMVertexPartition
            
            # Note: CPM usually requires passing resolution_parameter, Modularity doesn't necessarily
            if args.objective == "CPM":
                partition = leidenalg.find_partition(g, partition_type, resolution_parameter=args.resolution)
            else:
                partition = leidenalg.find_partition(g, partition_type)
            membership = partition.membership
        except ImportError:
            raise ImportError("The 'leidenalg' python package is required for older igraph versions. Install it using 'pip install leidenalg'.")

    num_communities = len(set(membership))
    print(f"Found {num_communities} communities.")

    print("Preparing output...")
    out_data = {
        "node_id": [idx_to_node[i] for i in range(num_nodes)],
        "cluster_id": membership
    }
    out_df = pd.DataFrame(out_data)
    
    out_df.to_csv(args.output, index=False)
    print(f"Output successfully written to {args.output}")

if __name__ == "__main__":
    main()
