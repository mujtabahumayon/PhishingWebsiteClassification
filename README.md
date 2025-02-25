# 🛡️ Phishing Websites Classification

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Pytorch](https://img.shields.io/badge/PyTorch-1.10%2B-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-0.24%2B-orange.svg)

## 📌 Project Overview
This project utilizes a **Feedforward Neural Network (SimpleNN)** to classify websites as either **phishing** or **legitimate**. The model is trained on the **Phishing Websites Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/327/phishing+websites).

## 📂 Dataset
- **Name:** Phishing Websites Dataset
- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/327/phishing+websites)
- **Instances:** 11,055
- **Features:** 30 (all integer type)
- **Task:** Binary classification (Phishing or Legitimate)

## 🛠️ Methodology
1. **Data Preprocessing**
   - Loading dataset
   - One-hot encoding categorical variables
   - Splitting into training & validation sets
2. **Exploratory Data Analysis (EDA)**
   - Correlation heatmap
   - Dataset visualization
3. **Model Architecture (SimpleNN)**
   - Input Layer: 30 features
   - Hidden Layers: 
     - **Layer 1:** 64 neurons, ReLU activation
     - **Layer 2:** 32 neurons, ReLU activation
   - Output Layer: 2 neurons (Binary Classification)
   - Loss Function: CrossEntropyLoss
   - Optimizer: Adam (learning rate = 0.001)
   - Regularization: Dropout layers
4. **Training & Validation**
   - 10 training cycles
   - Hyperparameter tuning for better performance

## 📊 Results
### Without Hyperparameter Tuning:
- **Validation Accuracy:** 93.14%

### With Hyperparameter Tuning:
- **Optimized Learning Rate:** 0.01
- **Epochs:** 20
- **Final Validation Accuracy:** **99.73%** 🎯

## 🚀 Implementation
The project is implemented using the following technologies:
- **Python 🐍**
- **PyTorch 🔥**
- **Scikit-learn 🤖**
- **Pandas & NumPy 📊**
- **Seaborn & Matplotlib 📈**

## 🔥 Challenges Faced
- Handling categorical variables efficiently
- Fine-tuning hyperparameters for optimal accuracy
- Ensuring model generalization to new, unseen data

## 🎯 Future Improvements
- **Feature Engineering:** Incorporate additional website features
- **Ensemble Models:** Combine multiple classifiers for better accuracy
- **Deployment:** Convert the model into a web app for real-world use

## 📜 References
- [Understanding Feedforward Neural Networks](https://learnopencv.com/understanding-feedforward-neural-networks/)
- [Correlation in Python](https://medium.com/@polanitzer/correlation-in-python-find-statistical-relationship-between-variables-bfeb323c16d6)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## 👨‍💻 Author
- **Mujtaba Humayon**   
  - **Course:** Deep Learning
  - **Project:** Final Project on Phishing Website Classification  

---

🌟 *If you find this project useful, consider giving it a ⭐ on GitHub!*
