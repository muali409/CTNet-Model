import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Function to evaluate the model on test data
def evaluate_model(model, test_mias_data, test_breakhis_data, test_labels):
    """
    Evaluates the CTNet model on test data.

    Parameters:
    model (tf.keras.Model): Trained CTNet model.
    test_mias_data (np.array): Test images from MIAS dataset.
    test_breakhis_data (np.array): Test images from BreakHis dataset.
    test_labels (np.array): True labels for test data.

    Returns:
    None: Prints evaluation metrics.
    """

    # Predict on test data
    y_pred_probs = model.predict([test_mias_data, test_breakhis_data])
    
    # Convert predicted probabilities to class labels
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(test_labels, axis=1)

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Compute and print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # Compute and print overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Example usage with test data (assuming data has been preprocessed):
# Assuming you have preprocessed test MIAS and BreakHis datasets in test_mias_data and test_breakhis_data
# and their corresponding labels in test_labels

# Load or define your trained model
# model = load_model('path_to_your_trained_model.h5') # Uncomment if loading a saved model

# Call evaluate_model function to test the model
evaluate_model(model, test_mias_data, test_breakhis_data, test_labels)
