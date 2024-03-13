# Model Evaluation with Streamlit
This is a Python application built with Streamlit for evaluating machine learning models on the Kannada MNIST dataset. The application allows users to select different models and PCA components to analyze their performance metrics such as precision, recall, and F1-score. Additionally, it provides visualizations of confusion matrices and ROC-AUC curves for each selected model.

Setup
Before running the application, ensure you have Python installed on your system. You can install the required libraries by running:

bash
Copy code
pip install -r requirements.txt
Running the Application
To run the application, execute the following command:

bash
Copy code
streamlit run app.py
This will start a local server, and you can access the application in your web browser at http://localhost:8501.

Overview of the Code
Libraries Used
streamlit: for building the web application interface.
numpy: for numerical computations.
matplotlib, seaborn: for data visualization.
scikit-learn: for machine learning models and evaluation metrics.
yellowbrick: for visualizing ROC-AUC curves.
Main Components
get_classifier(model): Function to retrieve the appropriate classifier based on the selected model.
calculate_metrics(y_true, y_pred): Function to calculate precision, recall, and F1-score.
display_confusion_matrix(y_test, y_pred, model): Function to display the confusion matrix.
display_roc_auc_curve(classifier, X_train_pca, y_train, X_test_pca, y_test, model): Function to display the ROC-AUC curve.
display_metrics_comparison(metrics_df): Function to display the metrics comparison table.
Dataset
The Kannada MNIST dataset is used for this evaluation. It consists of handwritten digits in the Kannada script, with a total of 60,000 training images and 10,000 test images.

Models
The following machine learning models are available for evaluation:

Decision Trees
Random Forest
Naive Bayes
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
How to Use
Use the sidebar to select models and PCA components.
View the evaluation metrics and visualizations for the selected models.
Contributors
Your Name
Contributor 1
Contributor 2
License
This project is licensed under the MIT License - see the LICENSE file for details.

Feel free to customize this README content according to your project's specific details and requirements.