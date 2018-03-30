import graphviz
tree_data = sklearn.tree.export_graphviz(dtc, out_file=None,
                         feature_names=xcolnames,
                         class_names=['churn', 'not churn'],
                         filled=True, rounded=True,
                         special_characters=True)
graph = graphviz.Source(tree_data)
graph
