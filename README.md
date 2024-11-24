# ML From Scratch

This repository contains implementations of Machine Learning algorithms and models from scratch. The purpose of this repository is to provide a simplistic form of these algorithms and match them to their respective libraries in Python.

## Table of Contents

**Statistical Models**

- [x] [**Linear Regression**][lin-reg] – Basic regression model, foundation for learning.
- [x] [**Logistic Regression**][logistic-reg] – Basic classification model.
- [x] [**Ridge and Lasso Regression**][ridge-lasso] – Regularized versions of Linear Regression.
- [x] [**K-Nearest Neighbors (KNN)**][knn] – Simple non-parametric model for classification/regression.
- [x] [**Naive Bayes**][naive-bayes] – Probabilistic model based on Bayes' theorem.
- [x] [**K-Means**][k-means] – Basic clustering algorithm.
- [ ] [**T-SNE**][t-sne] - t-Distributed Stochastic Neighbor Embedding, dimensionality reduction technique.
- [ ] **Principal Component Analysis (PCA)** – Dimensionality reduction.
- [ ] **Singular Value Decomposition (SVD)** – Matrix factorization for dimensionality reduction.
- [ ] **Decision Trees** – Simple interpretable model for classification/regression.
- [ ] **Random Forests** – Ensemble of Decision Trees, improves accuracy.
- [ ] **Support Vector Machines (SVM)** – Powerful classification/regression model.
- [ ] **XGBoost** – State-of-the-art boosting algorithm for classification/regression.
- [ ] **Bayesian Networks** – Probabilistic graphical model.
- [ ] **Markov Decision Processes (MDPs)** – Framework for modeling decisions, often used in RL.

**Neural Network based Models:**

- [ ] **Neural Networks (NNs)** – Foundation for deep learning, learning weights and activations.
- [ ] **Backpropagation** – Essential algorithm for training NNs.
- [ ] **Gradient Descent** – Optimization method used in NNs.
- [ ] **Convolutional Neural Networks (CNNs)** – Specialized NNs for image data.
- [ ] **Recurrent Neural Networks (RNNs)** – NNs for sequential data.
- [ ] **Long Short-Term Memory Networks (LSTMs)** – Improved RNNs, handling long sequences.
- [ ] **Gated Recurrent Units (GRUs)** – Another variant of RNNs, simpler than LSTMs.
- [ ] **Transformer Networks** – State-of-the-art model for sequential data (e.g., NLP).
- [ ] **Autoencoders** – NNs for unsupervised learning and dimensionality reduction.
- [ ] **Generative Adversarial Networks (GANs)** – NNs for generating new data.
- [ ] **Reinforcement Learning** – Learning to take actions in an environment.
- [ ] **Q-Learning** – Used in Reinforcement Learning; can involve neural nets.

[lin-reg]: ./01-linear-regression.ipynb
[logistic-reg]: ./02-logistic-regression.ipynb
[ridge-lasso]: ./03-ridge-lasso-regression.ipynb
[knn]: ./04-knn.ipynb
[naive-bayes]: ./05-naive-bayes.ipynb
[k-means]: ./06-k-means.ipynb
[t-sne]: ./07-t-sne.ipynb

## Installation

1. Clone the repository
2. Install the required dependencies
3. Run the desired notebook

## FAQs

#### Why are you doing this?

I am doing this to learn more about the inner workings of Machine Learning algorithms and models. I believe that by implementing these algorithms from scratch, I will have a better understanding of how they work and how they can be improved.

#### Why not just use libraries like `scikit-learn` or `tensorflow` or `pytorch`?

While libraries like `scikit-learn` and `tensorflow` are great for building Machine Learning models quickly, they abstract away a lot of the details of how these models work. By implementing these models from scratch, I can gain a deeper understanding of how they work and how they can be improved.

#### How do I know if my implementation is correct?

I will be comparing the results of my implementations to the results of the corresponding libraries in Python. If the results match, then I can be confident that my implementation is correct.

#### Can I use this in production?

Send me an email, I can tell you faster ways to get a headache :)

## Contributing

If you would like to contribute to this repository, please open an issue or a pull request.

## License

This repository is licensed under the MIT License. Do whatever you want with it!
