# Iris Perceptron Predictor

## About
This is an improved version of my Iris Perceptron. This version includes a **partial fit** function to allow additional learning with new data and utilizes **stochastic gradient descent** for an optimal learning curve. It also provides visualization features, allowing the user to see the data distribution and the learning curve via cost convergence.

This project is a simple AI that uses a Perceptron model to classify Iris flowers based on their **sepal length** and **petal length**. It uses the **Iris dataset** from the UCI Machine Learning Repository.

---

## Requirements
To run this project, you'll need to have the following installed on your local machine:

1. **Python 3.x**  
   Ensure you have Python 3.6 or higher installed. You can download it from [here](https://www.python.org/downloads/).

2. **Pip**  
   Pip comes pre-installed with Python, but if you don't have it, follow the instructions [here](https://pip.pypa.io/en/stable/installation/).

---

## Setting Up the Environment

### 1. Clone the Repository
If you haven't cloned the repository yet, do so by running:

```bash
git clone https://github.com/Cremdemout1/improved_perceptron.git
cd improved_perceptron
```

### 2. Create a Virtual Environment (Optional, but Recommended)
Creating a virtual environment ensures you don't interfere with your global Python environment. To create one, run:

```bash
../make_env.sh
```

Activate the virtual environment:

- **On Windows**:
  ```bash
  new_env\Scripts\activate
  ```
- **On macOS/Linux**:
  ```bash
  source new_env/bin/activate
  ```


### 4. Run the Script
Once your environment is set up, you can run the script:

```bash
python ml.py
```

This will load the Iris dataset, train the Perceptron model, and display decision boundary visualizations.

---

## How It Works
This project uses a **Perceptron** model, a fundamental binary classifier in machine learning. It works by learning a linear decision boundary between two classes based on input data.

1. **Dataset**: The Iris dataset contains features like sepal length and petal length for different species of Iris flowers. This project focuses on the first 100 samples, which belong to two species: **Iris-setosa** and **Iris-versicolor**.

2. **Preprocessing**: We select two features, **sepal length** and **petal length**, to predict whether a flower is **Iris-setosa** (represented by `-1`) or **Iris-versicolor** (represented by `1`).

3. **Training**: The Perceptron is trained using stochastic gradient descent (SGD) for optimal learning. The model is updated iteratively using a cost function that minimizes classification error.

4. **Visualization**: The model provides:
   - **Decision boundary plots** to show how the Perceptron classifies data.
   - **Cost convergence graphs** to illustrate the model's learning progress.

5. **Prediction**: Once trained, the Perceptron can classify new samples based on user input for sepal and petal length.

---

## Example Interaction

```bash
Please write 'exit' to leave and 'examine' to enter AI Iris predictor
examine
Enter sepal length of Iris: 5.1
Enter petal length of Iris: 1.4
Iris Setosa
```

---

## Features
- **Partial Fit**: Allows incremental learning with new data.
- **Stochastic Gradient Descent**: Optimized training process.
- **Graphical Analysis**: Decision boundary and cost convergence plots.
- **Simple User Input**: Interactive prediction system.

This implementation enhances the learning efficiency and visualization capabilities, making it a robust and adaptable Perceptron model.

---

## License
This project is open-source and available for modification and improvement.

# improved_perceptron
