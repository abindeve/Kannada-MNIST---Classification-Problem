import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
import seaborn as sns
from yellowbrick.classifier import ROCAUC
import pandas as pd

st.set_page_config(page_title="Model Evaluation with Streamlit", layout="wide", initial_sidebar_state="expanded")
st.set_option('deprecation.showPyplotGlobalUse', False)

def get_classifier(model):
    if model == 'Decision Trees':
        return DecisionTreeClassifier()
    elif model == 'Random Forest':
        return RandomForestClassifier()
    elif model == 'Naive Bayes':
        return GaussianNB()
    elif model == 'KNN':
        return KNeighborsClassifier()
    elif model == 'SVM':
        return SVC()

def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return precision, recall, f1

def display_confusion_matrix(y_test, y_pred, model):
    st.title(f"Confusion Matrix - {model}")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f'{model} - Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot()

def display_roc_auc_curve(classifier, X_train_pca, y_train, X_test_pca, y_test, model):
    st.title(f"ROC-AUC Curve - {model}")
    fig, ax = plt.subplots(figsize=(6, 4))
    visualizer = ROCAUC(classifier, classes=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], ax=ax)
    visualizer.fit(X_train_pca, y_train)
    visualizer.score(X_test_pca, y_test)
    st.pyplot(fig)

def display_metrics_comparison(metrics_df):
    st.subheader("Model Comparison Table")
    st.write(metrics_df)
def main():
    ds_dir = "D:\\final project\\Kannada_MNIST\\kANNADA_MNIST"
    X_train = np.load(os.path.join(ds_dir, 'X_kannada_MNIST_train.npz'))['arr_0']
    X_test = np.load(os.path.join(ds_dir, 'X_kannada_MNIST_test.npz'))['arr_0']
    y_train = np.load(os.path.join(ds_dir, 'y_kannada_MNIST_train.npz'))['arr_0']
    y_test = np.load(os.path.join(ds_dir, 'y_kannada_MNIST_test.npz'))['arr_0']

    st.title('Kannada MNIST - Classification Problem')

    # Model and PCA component selection
    model_names = ['Decision Trees', 'Random Forest', 'Naive Bayes', 'KNN', 'SVM']
    pca_components = [10, 15, 20, 25, 30]

    selected_models = st.sidebar.multiselect('Select Models', model_names, default=model_names)
    selected_pca_components = st.sidebar.multiselect('Select PCA Components', pca_components, default=[20])

    metrics_data = []  # List to store metrics for all models
    for selected_model in selected_models:
        for selected_component in selected_pca_components:

            # Perform PCA
            pca = PCA(n_components=selected_component)
            X_train_pca = pca.fit_transform(X_train.reshape(X_train.shape[0], -1))
            X_test_pca = pca.transform(X_test.reshape(X_test.shape[0], -1))

            # Train and evaluate selected model
            classifier = get_classifier(selected_model)
            classifier.fit(X_train_pca, y_train)
            y_pred = classifier.predict(X_test_pca)

            # Calculate metrics
            precision, recall, f1 = calculate_metrics(y_test, y_pred)

            # Display metrics
            st.write(f"Model: {selected_model}, PCA Components: {selected_component}")
            st.write(f"Precision: {precision}")
            st.write(f"Recall: {recall}")
            st.write(f"F1-Score: {f1}")

            # Display confusion matrix
            display_confusion_matrix(y_test, y_pred, selected_model)

            # Display ROC-AUC curve
            display_roc_auc_curve(classifier, X_train_pca, y_train, X_test_pca, y_test, selected_model)

            # Append metrics to the list
            metrics_data.append({
                'Model': selected_model,
                'PCA Components': selected_component,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1
            })

    # Create DataFrame from metrics data
    metrics_df = pd.DataFrame(metrics_data)

    # Display metrics comparison table
    display_metrics_comparison(metrics_df)

if __name__ == "__main__":
    main()
