import osmnx as ox
import networkx as nx

def combine_graphs(base_graph, *additional_graphs):
    """
    Combines multiple OSMnx graph objects into a base graph.
    
    Parameters:
    - base_graph: The main graph (OSMnx MultiDiGraph) to which other graphs will be added.
    - additional_graphs: One or more OSMnx MultiDiGraph objects to add to the base graph.
    
    Returns:
    - base_graph: The combined graph with nodes and edges from all input graphs.
    """
    # Iterate through each additional graph
    for graph in additional_graphs:
        # Add nodes from the additional graph to the base graph
        for node, node_data in graph.nodes(data=True):
            if node not in base_graph:
                base_graph.add_node(node, **node_data)
        
        # Add edges from the additional graph to the base graph
        for u, v, key, edge_data in graph.edges(keys=True, data=True):
            if not base_graph.has_edge(u, v, key):
                base_graph.add_edge(u, v, key=key, **edge_data)
    
    return base_graph

# Example usage:
G1 = ox.graph_from_place("Amsterdam", network_type='walk')
G2 = ox.graph_from_place("Zuidoost,Amsterdam", network_type='walk')
G3 = ox.graph_from_place("Diemen,Amsterdam", network_type='walk')

# Combine G2 and G3 into G1
combined_graph = combine_graphs(G1, G2, G3)

# Plot the combined graph
ox.plot_graph(combined_graph)