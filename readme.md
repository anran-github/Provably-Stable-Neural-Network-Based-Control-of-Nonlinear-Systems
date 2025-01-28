# Code Description

## Simulation Study
data collection:

    drone_datacollection.m

change $\theta$ value for the specific dataset.

NN model:

    network.py

NN model training:

    train_ip.py

## Drone Experiments
data collection:

    IP_datacollection_theta001.m

change $\theta$ value for the specific dataset.

NN model shares the same model as simulation.

training:

    train_drone.py

testing:

    test.py

## Paper Figures
You can regenerate plotted figures in our papear from folders:

    drone_experiments/NN_LQR

    PAPER_IMPORTANT

## Model Transition
Converting pytorch model to nnox format, using following code:

    tranfer_matlab.py

## Citation

    @article{LI2024109252,
    title = {Provably-stable neural network-based control of nonlinear systems},
    journal = {Engineering Applications of Artificial Intelligence},
    volume = {138},
    pages = {109252},
    year = {2024},
    issn = {0952-1976},
    doi = {https://doi.org/10.1016/j.engappai.2024.109252},
    url = {https://www.sciencedirect.com/science/article/pii/S0952197624014106},
    author = {Anran Li and John P. Swensen and Mehdi Hosseinzadeh}
    }