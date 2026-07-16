# KOMPARASI MODEL SUPPORT VECTOR MACHINE DAN INDOBERT PADA ASPECT BASED SENTIMENT ANALYSIS PENGGUNA APLIKASI E – COMMERCE : STUDI KASUS TOKOPEDIA & SHOPEE

 &amp; SVM to compare Model from Texting Review Indonesia E-Commerce
# 📋 Overview
E-commerce is one of the largest apps in the world, and even in Indonesia, it's a vital part of life for many people. E-commerce simplifies transactions and purchases of commercial goods. Therefore, e-commerce development is necessary through reviews of each app. The research uses various machine learning and deep learning methods as a comparison between prediction models and the accuracy of data. The model used is: 
- IndoBERT (Transformer)
- SVM (Support Vector Machine)

With these two models as a comparison, they can be used as the basis for the results of a prediction model created using Streamlit.
The research method used is CRISP-DM: starting from Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation and Deployment.

## Key Features
- 2 Model comparation of representative object text commentar
- 4 Aspect labeling data (Harga, Produk, Pengiriman, Sistem Aplikasi)
- Comprehensive evaluation using Recall, Precision, F1 Score, F1 Macro, Accuracy and Confusion Matrix and Accuracy aspect labeling
- Interactive Deployment - using Streamlit web app to made web prediction application to predict about texting commentar using 2 Models

# 🎯 Key Contributions
- Aspect Labeling : This aspect labeling uses the LLM model, namely MiniLM - L12, by looking for 1 review word that exactly matches 4 aspects.
- Modeling Data Text : Text data modeling was carried out using 2 models, namely IndoBERT and SVM, and before the data model was carried out, it was divided into 70:15:15 with a capacity of 70% training data, 15% validation data, and 15% test data.
- Comprehensive Evaluation : The evaluation results are seen from 3 aspects, namely Recall, Precision and F1 Score, which are the benchmarks for data balance and strength between training and testing data.

# 🏗️ Project Architecture / Research Workflow
The research follows the CRISP-DM methodology. The framework diagram (in English) is shown below
<img width="9088" height="3012" alt="image" src="https://github.com/user-attachments/assets/bb80ad12-1205-4193-8685-9b79e5ff450a" />

The same workflow : 
<img width="6176" height="4448" alt="image" src="https://github.com/user-attachments/assets/1d37287e-c738-41ee-91d5-81f5bc2555f2" />

# 🚀 Installation
Python 3.9+
PyTorch 2.x and Torchvision (matching CUDA build)
Ultralytics (YOLOv8)
Streamlit (for the web app)
OpenCV, Pillow, NumPy
(training/evaluation) Matplotlib, Seaborn, scikit-learn, timm
A CUDA-capable GPU is recommended for training (this research used Google Colab with an NVIDIA Tesla T4)
