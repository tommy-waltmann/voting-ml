### Hyperparameter Experiment

I ran the `optimal_params.py` script a bunch of times with different restrictions on
the two hyperparameters `max_leaf_nodes` and `max_depth` and the resultant graphs are
in this directory. I did this to help determine which restrictions on the hyperparameter
search we will need to have in order to have an interpretable graph output.

## Experiment Details

I did two experiments, one on the `max_leaf_nodes` parameter, and another on the `max_depth`
parameter. In each experiment, I fixed the value of one parameter and let the other be
whatever it wanted to be. The results are in this directory with files named according to the
fixed value of the experiment parameter.

## Results

When I fix `max_depth`, the number of leaf nodes grows very quickly, which creates problems
interpreting the graph, even at relatively small values like `max_depth=4`

When I fix `max_leaf_nodes`, there tends to be more than the mimimum depth, but the graph still
stays interpretable, problems still do arise when there gets to be a lot of leaf nodes.

I believe the parameters `max_depth=5` and `max_leaf_nodes=10` are the maximum values we should
allow those parameters to take in order to interpret the learned decision tree.

## File names

The pngs in this directory are named according to the parameter that was fixed, and the value it
was fixed at.
- mln = `max_leaf_nodes`
- md = `max_depth`

