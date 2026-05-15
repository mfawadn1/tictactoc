import pickle
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from data_preprocessing import load_and_preprocess_data

def train_svm():
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_data()
    
    print("Training SVM model...")
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/svm_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    return acc, cm

if __name__ == "__main__":
    acc, cm = train_svm()
    print(f"SVM Accuracy: {acc:.4f}")
    print(f"Confusion Matrix:\n{cm}")
