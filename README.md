# multiagentcontrol
Multiagentcontrol provides a framework for multi-agent control systems simulation, with support of CPU parallelisation to run multiple instances of multi-agent systems simultaneously.

## Dependencies
The following packages are required:
- numpy
- scipy
- matplotlib
- networkx
- pathos (only for parallelisation)

## Installation
Clone the repository to your computer

```
git clone https://github.com/elvistkf/multiagentcontrol.git
```

Navigate into the cloned repository folder

```
cd multiagentcontrol
```

Install the package via pip

```
python -m pip install .
```

or

```
pip install .
```

It is highly recommended to install the package inside a virtual environment via ``conda`` or ``venv``.

## References
This work is based on the following literature:

[1] K. F. E. Tsang, J. Wu and L. Shi, "Distributed Optimisation with Stochastic Event-Triggered Multi-Agent Control Algorithm", *IEEE Conference on Decision and Control*, pp. 6222-6227, 2020.

[2] X. Yi, K. Liu, D. V. Dimarogonas and K. H. Johansson, "Dynamic Event-Triggered and Self-Triggered Control for Multi-agent Systems", *IEEE Transactions on Automatic Control*, vol. 64, no. 8, pp. 3300-3307, 2019.

[3] W. Du, X. Yi, J. George, K. H. Johansson and T. Yang, "Distributed Optimization with Dynamic Event-Triggered Mechanisms", *IEEE Conference on Decision and Control*, pp. 969-974, 2018.

[4] K. F. E. Tsang, J. Wu and L. Shi, "Zeno-Free Stochastic Distributed Event-Triggered Consensus Control for Multi-Agnet Systems", *American Control Conference*, pp. 778-783, 2018.
