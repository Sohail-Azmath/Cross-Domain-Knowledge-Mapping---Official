import networkx as nx
from pyvis.network import Network

# -- STEP 1: Build the Graph (same as your concept) --

G = nx.MultiDiGraph()

domains = {
    'Computer Science': {
        'color': '#3498db',
        'concepts': [
            {'id': 'CS1', 'label': 'Algorithm'},
            {'id': 'CS2', 'label': 'Data Structure'},
            {'id': 'CS3', 'label': 'Function'},
            {'id': 'CS4', 'label': 'Recursion'},
        ]
    },
    'Cooking': {
        'color': '#e74c3c',
        'concepts': [
            {'id': 'CK1', 'label': 'Recipe'},
            {'id': 'CK2', 'label': 'Ingredient'},
            {'id': 'CK3', 'label': 'Cooking Method'},
            {'id': 'CK4', 'label': 'Measurement'},
        ]
    },
    'Project Management': {
        'color': '#2ecc71',
        'concepts': [
            {'id': 'PM1', 'label': 'Project Plan'},
            {'id': 'PM2', 'label': 'Task'},
            {'id': 'PM3', 'label': 'Timeline'},
            {'id': 'PM4', 'label': 'Resource'},
        ]
    },
    'Common': {
        'color': '#95a5a6',
        'concepts': [
            {'id': 'CM1', 'label': 'Process'},
            {'id': 'CM2', 'label': 'Sequence'},
        ]
    }
}

relationships = [
    ('CS1', 'CS2', 'uses', 'solid'),
    ('CS2', 'CS3', 'implements', 'solid'),
    ('CS3', 'CS4', 'enables', 'solid'),
    ('CK1', 'CK2', 'requires', 'solid'),
    ('CK2', 'CK3', 'used_in', 'solid'),
    ('CK3', 'CK4', 'needs', 'solid'),
    ('PM1', 'PM2', 'contains', 'solid'),
    ('PM2', 'PM3', 'scheduled_in', 'solid'),
    ('PM2', 'PM4', 'uses', 'solid'),
    # Cross-domain
    ('CS1', 'CK1', 'analogous_to', 'dashed'),
    ('CS2', 'CK2', 'similar_to', 'dashed'),
    ('CS3', 'CK3', 'maps_to', 'dashed'),
    ('PM1', 'CS1', 'requires', 'dashed'),
    ('PM2', 'CK2', 'analogous_to', 'dashed'),
    ('CM1', 'CS1', 'describes', 'dotted'),
    ('CM1', 'CK1', 'describes', 'dotted'),
    ('CM2', 'CS2', 'describes', 'dotted'),
    ('CM2', 'CK2', 'describes', 'dotted'),
]

# Add nodes with color and label
node_color_dict = {}
node_label_dict = {}
for domain, info in domains.items():
    for concept in info['concepts']:
        G.add_node(concept['id'], label=concept['label'], color=info['color'])
        node_color_dict[concept['id']] = info['color']
        node_label_dict[concept['id']] = concept['label']

# Add edges with style and relation label
for src, tgt, rel, style in relationships:
    G.add_edge(src, tgt, label=rel, style=style)

# -- STEP 2: Use PyVis for interactive HTML export --

net = Network(height='850px', width='100%', directed=True)
net.barnes_hut(gravity=-80000)  # better separation

for node, attr in G.nodes(data=True):
    net.add_node(node, label=attr['label'], color=attr['color'])

for src, tgt, attr in G.edges(data=True):
    dash = True if attr['style'] in ['dashed', 'dotted'] else False
    width = 2 if attr['style'] == 'solid' else 2.5
    net.add_edge(src, tgt, label=attr['label'], width=width, physics=True, dashes=dash)

net.set_options("""
const options = {
  "edges": {
    "smooth": {
      "type": "cubicBezier"
    },
    "arrows": {
      "to": {"enabled": true}
    }
  }
}
""")

# -- STEP 3: Save interactive HTML file --
net.show("small_knowledge_graph.html", notebook=False)
print("✓ Interactive knowledge graph saved as 'cross_domain_knowledge_graph.html'")

# -- STEP 4: Print ALL interconnected component node clusters --
subgraphs = list(nx.connected_components(G.to_undirected()))
for idx, cluster in enumerate(subgraphs):
    print(f"Cluster {idx + 1}: Nodes - {sorted(cluster)}")
