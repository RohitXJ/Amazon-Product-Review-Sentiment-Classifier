# Review Sentiment Analyzer

This project is part of a broader initiative to explore both supervised and unsupervised learning approaches in machine learning. This repository contains the implementation of a Sentiment Analysis pipeline inside a Jupyter Notebook and the final trained models.

## 📌 Project Overview

The goal of this project is to classify customer reviews into one of three sentiment categories:

* **Positive**
* **Negative**
* **Neutral**

The analysis was conducted in two phases: an unsupervised exploration and a supervised training pipeline.

## 🧠 Learning Scope

### 🔍 Phase 1: Unsupervised Learning

We began with an unlabeled dataset to uncover hidden structures and patterns using clustering techniques. This helped us better understand the distribution and variety of sentiments present.

* Data Cleaning and Preprocessing
* Vectorization using TF-IDF
* Dimensionality Reduction
* K-Means Clustering to identify possible sentiment groups

### 🧠 Phase 2: Supervised Learning

Once labels were prepared, we trained multiple supervised models to identify the best-performing one for the sentiment classification task.

* Data preprocessing (text normalization, cleaning)
* TF-IDF vectorization
* Model experimentation with:

  * Logistic Regression
  * Random Forest
  * Naive Bayes
  * Linear Support Vector Classifier (SVC)
* Linear SVC was selected as the best model based on performance metrics

## 🔗 Dataset

The dataset used is publicly available:

* [Amazon Product Reviews Dataset](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews)

## 🧪 Model Artifacts

The final repository includes:

* A trained **Linear SVC** model (`model.pkl`)
* A **TF-IDF Vectorizer** (`vectorizer.pkl`)

## 📁 Repository Structure

```
review-sentiment-analyzer/
├── Sentiment_Analysis_Notebook.ipynb  # Complete ML pipeline notebook
├── linear_svc_model.pkl                          # Final Linear SVC model
├── tfidf_vectorizer.pkl                     # TF-IDF vectorizer
├── README.md                          # This file
```

## 💡 Skills Demonstrated

* Text preprocessing (cleaning, vectorization)
* Clustering (unsupervised learning)
* Supervised ML model training and evaluation
* Model selection and comparison

## 🛠️ Tools & Libraries

* Python
* Scikit-learn
* Pandas
* Numpy
* Matplotlib / Seaborn (for EDA)

## 👨‍💻 Author

**Rohit Gomes**

## Check out my deployed model: [My Model][huggingface_link]

[huggingface_link]: https://huggingface.co/spaces/RJx334/review-sentiment-analyzer

---

*This project was developed to strengthen core ML skills by combining exploratory analysis with practical unsupervised and supervised learning implementation.*
