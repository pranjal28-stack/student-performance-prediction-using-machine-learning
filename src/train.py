import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib

# Import the ML models from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Import evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def train_and_evaluate():
    # 1. Load the preprocessed numpy arrays from Phase 3
    try:
        data_assets = np.load('data/processed/split_data.npz')
    except FileNotFoundError:
        print("\n[ERROR] 'data/processed/split_data.npz' not found!")
        print("Please run 'python preprocess.py' first to generate the processed data assets.\n")
        return

    X_train = data_assets['X_train']
    X_test = data_assets['X_test']
    y_train = data_assets['y_train']
    y_test = data_assets['y_test']
    
    # 2. Initialize the three models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(max_depth=4, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    }
    
    # Create a dictionary to hold our evaluation metrics for comparison
    performance_metrics = {
        'Model': [],
        'Accuracy': [],
        'Precision': [],
        'Recall': [],
        'F1 Score': []
    }
    
    # 3. Loop through each model, train, and evaluate
    for model_name, model_obj in models.items():
        print(f"Training {model_name}...")
        
        # Train the model (Andrew Ng's "Fit" step)
        model_obj.fit(X_train, y_train)
        
        # Make predictions on the unseen test data
        y_pred = model_obj.predict(X_test)
        
        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Store metrics in our dictionary
        performance_metrics['Model'].append(model_name)
        performance_metrics['Accuracy'].append(acc)
        performance_metrics['Precision'].append(prec)
        performance_metrics['Recall'].append(rec)
        performance_metrics['F1 Score'].append(f1)
        
        # Compute and display the Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"--- {model_name} Evaluation ---")
        print(f"Accuracy:  {acc:.4f} | Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f} | F1 Score:  {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}\n")
        
        # Save the trained model file to disk so we can use it later
        os.makedirs('models', exist_ok=True)
        joblib.dump(model_obj, f'models/{model_name.lower().replace(" ", "_")}_model.joblib')
        
    # 4. Create a comparison DataFrame
    metrics_df = pd.DataFrame(performance_metrics)
    os.makedirs('results', exist_ok=True)
    metrics_df.to_csv('results/model_comparison.csv', index=False)
    
    # 5. Visualization: Model Comparison Bar Chart
    plt.figure(figsize=(10, 6))
    
    # Setting up bar dimensions
    x_indexes = np.arange(len(metrics_df['Model']))
    width = 0.18
    
    # Plotting bars side-by-side for comparison
    plt.bar(x_indexes - width*1.5, metrics_df['Accuracy'], width=width, label='Accuracy', color='#4A90E2')
    plt.bar(x_indexes - width*0.5, metrics_df['Precision'], width=width, label='Precision', color='#50E3C2')
    plt.bar(x_indexes + width*0.5, metrics_df['Recall'], width=width, label='Recall', color='#F5A623')
    plt.bar(x_indexes + width*1.5, metrics_df['F1 Score'], width=width, label='F1 Score', color='#E2849A')
    
    plt.xticks(ticks=x_indexes, labels=metrics_df['Model'])
    plt.title('Model Performance Comparison')
    plt.ylabel('Score (0.0 to 1.0)')
    plt.ylim(0, 1.1)
    plt.legend(loc='lower right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png')
    plt.close()
    print("[Saved] Performance comparison plot saved to 'results/model_comparison.png'")

if __name__ == "__main__":
    train_and_evaluate()