This folder contains test with code using the same pseudopotential set.

Abinit, CASTEP and QE are computed using PseudoDojo v0.4 'standard' scalar-relativistic  potentials (SR) with 'high' cut-off stringency

Abacus is also computed using this set with the protocol specified in the aiida-abacus and aiida-common-workflows, but the standard cut-off stringency is used (normal).

TODO: 
1. Check the protocol settings with the developer
2. Rerun with 'high' cut off stringency (same potentials). The method in the AbacusBaseWorkChain should be updated to allow setting cut off stringency from the protocol dictionary.
