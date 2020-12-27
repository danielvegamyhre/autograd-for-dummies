import os 
from autograd.engine import Scalar
from autograd.visualize import draw_graph

def test_draw_graph():
    # initialize inputs
    a = Scalar(1.5)
    b = Scalar(-4.0)

    # forward pass
    c = a**3 / 5
    d = c + (b**2).relu()

    # backpropagation
    d.backward()

    # visualize computational graph
    graph = draw_graph(d)
    pdf = graph.view()

    assert isinstance(pdf, str), 'graph.view() returned None!'
    assert pdf.endswith('.png'), 'graph.view() did not produce PNG!'

    # cleanup
    os.remove('Digraph.gv') or 'cleanup of Digraph.gv failed!'
    os.remove('Digraph.gv.png') or 'cleanup of Digraph.gv.png failed!'
