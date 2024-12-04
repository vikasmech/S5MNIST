# MNIST CNN Classifier with CI/CD Pipeline

This project implements a Convolutional Neural Network (CNN) for MNIST digit classification with an automated CI/CD pipeline using GitHub Actions. The model achieves >95% accuracy on the test set with just one epoch of training.

## Model Architecture

The CNN architecture consists of:
- 4 Convolutional layers with increasing channel dimensions (1→10→16→20→10)
- 2 MaxPooling layers for spatial dimension reduction
- ReLU activation functions
- Final fully connected layer mapping to 10 classes
- Total parameters: <25,000

## Project Structure

.
├── model/
│ ├── init.py # Makes model directory a Python package
│ └── network.py # Contains CNN architecture definition
├── tests/
│ └── test_model.py # Contains model tests and validations
├── .github/
│ └── workflows/
│ └── ml-pipeline.yml # GitHub Actions CI/CD configuration
├── train.py # Script for model training
├── requirements.txt # Project dependencies
├── .gitignore # Specifies which files Git should ignore
└── README.md # Project documentation

## CI/CD Pipeline

The GitHub Actions workflow automatically:
1. Sets up Python 3.8 environment
2. Installs CPU-only versions of PyTorch and other dependencies
3. Trains the model for one epoch
4. Runs all tests to validate model performance
5. Archives the trained model as an artifact

The pipeline is triggered on every push to the repository. You can find the trained models in the Actions tab of your GitHub repository under artifacts.

## Model Performance

With just one epoch of training, the model:
- Achieves >95% accuracy on the MNIST test set
- Uses less than 25,000 parameters
- Processes standard MNIST images (28x28 pixels)
- Outputs probabilities for 10 digit classes (0-9)