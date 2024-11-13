import networkx as nx
import numpy as np

def plot_graph(graph, center, scale = 2, kwargs = {}):
  layout = nx.circular_layout(graph.predecessors(center), scale)
  layout[center] = np.array([0, 0])
  for x in graph.predecessors(center):
    layout.update(nx.circular_layout(graph.predecessors(x), scale = (scale // 2), center = layout[x]))
  return nx.draw(graph, layout, **kwargs)