### About this project 

This project has a couple of components regarding applying network centralities to the CommunityFitNet network benchmarking dataset.

The script domirank.py contains code implemented from reading the DomiRank paper (M. Engsig, A. Tejedor, Y. Moreno, E. Foufoula-Georgiou, C. Kasmi,
"DomiRank Centrality: revealing structural fragility of complex networks via node dominance."
https://arxiv.org/abs/2305.09589).

Completed: an analytical solution to domirank centrality.
Future Work: a power-iteration implementation of domirank. Additionally, the tests 

The dataLoader.py script applies a handful of network centrality methods to the networks in the CommunityFitNet_updated.pickle file.

Completed: domirank, degree, eigenvector and pagerank centrality have all been implemented.
Future Work: Improve code generalizability and add more centrality measurements. Additionally, SpringRank appears not to be taking advantage of edge weights and therefore not generating appropriate results.

The analysis_notebook.ipynb contains code to read in the CommunityFitNet_updated_centralities.pickle file and generate visualizations from it. It also contains some examples of tests for Domirank as well as work in progress for the power-iteration implementation.