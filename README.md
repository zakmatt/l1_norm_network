# RNN for Computing the L1 Norm of Variable-Length Sequences
This repository contains a PyTorch implementation of a custom Recurrent Neural Network (RNN) designed to compute the L1 norm of a variable-length input sequence of real-valued numbers without explicitly using the L1 norm operation. The network uses ReLU activations, negation, and standard operations like sum and multiplication to achieve this task.

## Project Structure
- l1_norm_model.ipynb: The l1_norm_model Jupyter Notebook containing the implementation and training of the RNN model.
- Pipfile: The Pipenv file that lists the dependencies required to run the project.

## Installation
To set up the environment and run the Jupyter Notebook, follow these steps:

1. **Clone the Repository:**
```bash
git clone https://github.com/zakmatt/l1_norm_network
cd l1_norm_network
```

2. **Install Pipenv:**  
If you don't have Pipenv installed, you can install it using pip:
```bash
pip install pipenv
```

3. **Install the Dependencies:**  
Use Pipenv to install all the required packages listed in the Pipfile:
```bash
pipenv install
```

4. **Activate the Environment:**  
```bash
pipenv shell
```

5. **Run Jupyter Lab:**  
Launch Jupyter Lab to view and execute the notebook:
```bash
jupyter lab
```

## Running the Experiment
The Jupyter Notebook l1_norm_model.ipynb is the main entry point of the project. It includes the following:

1. #### Environment Setup:

- The SQLite database mlflow.db is created for tracking experiments using MLflow.
- The MLflow tracking URI is set to use the SQLite backend.

2. #### Model Definition:

- A custom RNN cell (CustomRNNCell) is implemented, which processes the input sequence and computes the L1 norm without explicitly using the L1 norm operation.
- The RNN model (RNN) uses this custom cell to handle variable-length sequences.

3. #### Training:

- The model is trained using randomly generated sequences. The goal is for the RNN's weights to converge to values that correctly compute the L1 norm.
- The training process is logged using MLflow, allowing for the tracking of parameters, metrics, and model artifacts.
T
4. #### Testing:

- After training, the model is tested on a new sequence to verify its performance.

## Experiment Tracking with MLflow
All experiments, including hyperparameters, loss metrics, and model weights, are tracked using MLflow. The experiment data is stored in the mlflow.db SQLite database.

### Running MLflow UI
To visualize the experiment results, run the MLflow UI with the following command:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
The UI will be available at http://localhost:5000, where you can explore the logged experiments.

## Key Concepts and Observations

### Problem Breakdown
- **Handling Variable-Length Input:** Achieved using an RNN architecture.
- **Handling Different Signs:** Input values are transformed to their absolute values using a combination of ReLU activation and negation.
- **Training Objective:** The goal is to achieve an RNN cell with weights that converge to 1 (or an identity matrix) to accurately compute the L1 norm.

### Important Findings
- Weight Initialization: Initializing weights with negative values can lead to non-convergence due to the behavior of ReLU, which outputs zero for negative inputs, resulting in information loss and dead neurons.
- Bias Impact: The behavior of the model can vary significantly depending on the initialization of weights and biases. Proper initialization (e.g., positive weights) is crucial for ensuring the model converges.

## Conclusion
This project demonstrates the construction and training of an RNN to compute the L1 norm of a sequence, with a focus on understanding the impact of weight initialization and activation functions on the model's ability to converge. The use of MLflow for experiment tracking provides valuable insights into the training process and helps identify the conditions under which the model performs optimally.

## Final Thoughts
The key to successful training in this scenario is to ensure that the weights are initialized appropriately to avoid issues with ReLU activation. This approach can be adapted to other problems where similar challenges exist.