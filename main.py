from svm_training import train_svm
from dt_training import train_dt
from knn_training import train_knn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_training_pipeline():
    print("=== Tic Tac Toe ML Training Pipeline ===\n")
    
    results = {}
    
    # Run SVM
    svm_acc, svm_cm = train_svm()
    results["SVM"] = {"accuracy": svm_acc, "cm": svm_cm, "path": "models/svm_model.pkl"}
    
    # Run Decision Tree
    dt_acc, dt_cm = train_dt()
    results["Decision Tree"] = {"accuracy": dt_acc, "cm": dt_cm, "path": "models/dt_model.pkl"}
    
    # Run KNN
    knn_acc, knn_cm = train_knn()
    results["KNN"] = {"accuracy": knn_acc, "cm": knn_cm, "path": "models/knn_model.pkl"}
    
    print("\n" + "="*40)
    print("FINAL RESULTS SUMMARY")
    print("="*40)
    
    best_model = ""
    max_acc = 0
    best_path = ""
    
    for name, data in results.items():
        print(f"\nModel: {name}")
        print(f"Accuracy: {data['accuracy']:.4f}")
        print(f"Confusion Matrix:\n{data['cm']}")
        
        if data['accuracy'] > max_acc:
            max_acc = data['accuracy']
            best_model = name
            best_path = data['path']
            
    print("\n" + "="*40)
    print(f"WINNER: {best_model} with {max_acc:.4f} accuracy")
    print("All models have been saved in the 'models/' folder.")
    print("="*40)
    
    # ------------------ PLOTTING ------------------
    os.makedirs("plots", exist_ok=True)
    
    # 1. Accuracy Chart
    model_names = list(results.keys())
    accuracies = [data['accuracy'] for data in results.values()]
    
    plt.figure(figsize=(8, 5))
    bars = plt.bar(model_names, accuracies, color=['#4ade80', '#60a5fa', '#f472b6'])
    plt.ylim(0, 1.1)
    plt.title('Model Accuracy Comparison', fontsize=14)
    plt.ylabel('Accuracy', fontsize=12)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', fontsize=11)
    plt.tight_layout()
    plt.savefig('plots/accuracy_comparison.png')
    plt.close()
    
    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (name, data) in zip(axes, results.items()):
        sns.heatmap(data['cm'], annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title(f'{name} Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrices.png')
    plt.close()
    print("Plots updated in 'plots/' directory.")
    # ----------------------------------------------
    
    return best_model, max_acc, best_path

def main():
    run_training_pipeline()

if __name__ == "__main__":
    main()