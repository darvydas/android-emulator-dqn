import graphviz

def visualize_q_tree_graphviz(root_node, q_values_for_state, get_node_features, filename="q_tree_visualization"):
    """
    Visualizes the XML tree with Q-values using graphviz.

    Args:
        root_node: The root node of the lxml tree.
        q_values_for_state: DQNAgent instance state value function.
        get_node_features: Function to extract features from an lxml node.
        filename: Name of the output graph file (e.g., "q_tree_visualization.pdf").
    """

    dot = graphviz.Digraph(comment='XML Tree with Q-values', format='pdf') # You can change format to 'png', 'svg', etc.
    node_id_counter = 0 # To ensure unique node IDs

    def add_node_recursively(xml_node, parent_graphviz_node=None):
        nonlocal node_id_counter # Allow modification of the counter

        node_features = get_node_features(xml_node)
        q_values = q_values_for_state(node_features)

        node_label = f"<{xml_node.tag}>"
        if xml_node.text and xml_node.text.strip():
            node_label += f"\\nText: '{xml_node.text.strip()}'" # Use \\n for new lines in labels

        q_value_labels = "\\nQ-values:\\n" # For multi-line label
        action_names = ["Child", "Next", "Prev", "Parent", "Noop"] # Action names for labels
        for i, q_value in enumerate(q_values):
            q_value_labels += f"{action_names[i]}: {q_value:.2f}\\n" # Format Q-values

        graphviz_node_name = f"node_{node_id_counter}" # Unique node ID
        dot.node(graphviz_node_name, label=node_label + q_value_labels, shape='box') # 'box' shape for nodes
        node_id_counter += 1

        if parent_graphviz_node:
            dot.edge(parent_graphviz_node, graphviz_node_name) # Edge from parent to child

        for child in xml_node:
            add_node_recursively(child, graphviz_node_name) # Recursive call for children

    add_node_recursively(root_node) # Start recursion from the root
    dot.render(filename, view=False) # Generate PDF (or other format) and don't immediately open

    print(f"Q-value tree visualization saved to {filename}.pdf")