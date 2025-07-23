################################################################################
# Phase 0: Import Libraries
################################################################################
print("Phase 0: Importing Libraries...")
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from collections import OrderedDict # For ordered model weights

import kagglehub

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPClassifier # Multi-layer Perceptron
from imblearn.over_sampling import SMOTE

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, precision_recall_curve, auc,
    log_loss
)

# Suppress warnings (especially from sklearn about convergence for MLP)
import warnings
warnings.filterwarnings('ignore')

print("✅ Libraries imported successfully.")
print("-" * 60)

################################################################################
# Phase 1: Load Data and Initial Exploration
################################################################################
print("Phase 1: Loading Data and Initial Exploration...")

dataset_name = "fedesoriano/heart-failure-prediction"
csv_file_name = "heart.csv" # As per dataset content

print(f"Attempting to download dataset '{dataset_name}' from Kaggle Hub...")
try:
    dataset_path_dir = kagglehub.dataset_download(dataset_name)
    csv_file_path = os.path.join(dataset_path_dir, csv_file_name)
    print(f"Dataset downloaded to directory: {dataset_path_dir}")
    print(f"Full path to CSV: {csv_file_path}")

except Exception as e:
    print(f"Error downloading dataset from Kaggle Hub: {e}")
    print("Please ensure you have set up your Kaggle API token (~/.kaggle/kaggle.json).")
    print("You might need to run 'pip install kaggle kagglehub'.")
    exit()

if not os.path.exists(csv_file_path):
    print(f"Error: File not found at {csv_file_path} after Kaggle Hub download attempt.")
    print("Please check the download path and file name.")
    exit()

try:
    df = pd.read_csv(csv_file_path)
except UnicodeDecodeError:
    print("UTF-8 decoding failed, trying ISO-8859-1...")
    df = pd.read_csv(csv_file_path, encoding='ISO-8859-1')

print("✅ Dataset loaded successfully.")
print("\nFirst 5 records:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nDescriptive Statistics:")
print(df.describe())

print("\nClass Distribution (0: No Heart Disease, 1: Heart Disease):")
print(df['HeartDisease'].value_counts())
print(
    f"Heart Disease prevalence: "
    f"{df['HeartDisease'].value_counts()[1] / len(df) * 100:.2f}%"
)
print("-" * 60)

################################################################################
# Phase 2: Data Preprocessing
################################################################################
print("Phase 2: Data Preprocessing...")

# Define features (X) and target (y)
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Identify categorical and numerical columns
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
categorical_cols = X.select_dtypes(include='object').columns.tolist()

print(f"Numerical columns: {numerical_cols}")
print(f"Categorical columns: {categorical_cols}")

# Preprocessing Pipelines
# Numerical features: Standard Scaling
# Categorical features: One-Hot Encoding
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Split data into global training and testing sets (80% train, 20% test)
# This test set will be used for global model evaluation during FL rounds
X_train_global, X_test_global, y_train_global, y_test_global = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(
    f"Global training set shape: {X_train_global.shape}, "
    f"Global test set shape: {X_test_global.shape}"
)
print(
    f"Global training class distribution:\n"
    f"{y_train_global.value_counts(normalize=True)}"
)
print(
    f"Global test class distribution:\n"
    f"{y_test_global.value_counts(normalize=True)}"
)

# --- Apply Preprocessing to Global Training and Test Data ---
print("\nApplying preprocessing to global training and test data...")
# Fit preprocessor on training data and transform both train and test
X_train_processed = preprocessor.fit_transform(X_train_global)
X_test_processed = preprocessor.transform(X_test_global)

# Get feature names after one-hot encoding for transformed data
feature_names_processed = (
    numerical_cols
    + list(preprocessor.named_transformers_['cat'].get_feature_names_out(
        categorical_cols
    ))
)

# Convert processed data back to DataFrame for SMOTE and client distribution
X_train_processed_df = pd.DataFrame(
    X_train_processed, columns=feature_names_processed
)
X_test_processed_df = pd.DataFrame(
    X_test_processed, columns=feature_names_processed
)


# --- Handling Class Imbalance using SMOTE on Global Training Data ---
# Applying SMOTE to the global training data before distributing to clients
# simplifies the simulation by ensuring clients have balanced training data.
print("\nApplying SMOTE to the global training data...")
smote = SMOTE(random_state=42)
X_train_resampled_global, y_train_resampled_global = smote.fit_resample(
    X_train_processed_df, y_train_global
)

print("Class distribution in original global training data:")
print(y_train_global.value_counts())
print("Class distribution after SMOTE on global training data:")
print(y_train_resampled_global.value_counts())

print("✅ Data preprocessing complete.")
print("-" * 60)

################################################################################
# Phase 3: Federated Learning Setup
################################################################################
print("Phase 3: Federated Learning Setup...")

# --- Configuration for Federated Learning ---
NUM_CLIENTS = 5
NUM_COMMUNICATION_ROUNDS = 50
CLIENT_LOCAL_EPOCHS = 5
RANDOM_STATE = 42

print(f"Number of clients: {NUM_CLIENTS}")
print(f"Number of communication rounds: {NUM_COMMUNICATION_ROUNDS}")
print(f"Client local epochs per round: {CLIENT_LOCAL_EPOCHS}")

# --- 3.1 Data Partitioning for Clients ---
print(f"\nPartitioning data into {NUM_CLIENTS} client datasets...")
X_train_clients = []
y_train_clients = []

# To ensure stratified sampling per client
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=NUM_CLIENTS, shuffle=True, random_state=RANDOM_STATE)

# We'll use the indices to split the globally resampled data
client_idx = 0
for train_index, _ in skf.split(X_train_resampled_global, y_train_resampled_global):
    X_train_clients.append(X_train_resampled_global.iloc[train_index])
    y_train_clients.append(y_train_resampled_global.iloc[train_index])
    print(
        f"Client {client_idx+1} local data shape: "
        f"{X_train_clients[-1].shape} / {y_train_clients[-1].shape}"
    )
    client_idx += 1

print("✅ Data partitioned successfully.")

# --- 3.2 Define Client and Server Classes for Federated Learning ---

class Client:
    def __init__(self, client_id, X_local, y_local, model_type, random_state):
        self.client_id = client_id
        self.X_local = X_local
        self.y_local = y_local
        self.model_type = model_type
        self.local_model = self._initialize_model(
            model_type, random_state, X_local.shape[1]
        ) # Pass feature count for dummy fit
        print(f"Client {self.client_id} initialized with "
              f"{len(self.X_local)} samples.")

    def _initialize_model(self, model_type, random_state, feature_count):
        if model_type == 'MLPClassifier':
            model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500, # Max iterations for the MLP, adjusted for overall training
                random_state=random_state,
                early_stopping=False, # For consistent training across rounds
                solver='adam',
                activation='relu',
                warm_start=True # Crucial for federated learning to continue training
            )
            # Dummy fit to initialize coefs_ and intercepts_ for MLPClassifier
            # MLPClassifier needs to be fitted at least once to have coefs_ and intercepts_
            dummy_X = np.random.rand(2, feature_count)
            dummy_y = np.array([0, 1])
            model.fit(dummy_X, dummy_y)
            return model
        else:
            raise ValueError("Unsupported model type for client.")

    def get_model_weights(self):
        """Returns the current weights/coefficients of the local model."""
        if self.model_type == 'MLPClassifier':
            if hasattr(self.local_model, 'coefs_'):
                return {
                    'coefs': self.local_model.coefs_,
                    'intercepts': self.local_model.intercepts_
                }
        return None # Fallback if weights aren't initialized or model type is unknown

    def set_model_weights(self, global_weights):
        """Sets the local model's weights to the global weights."""
        if self.model_type == 'MLPClassifier':
            # Ensure the structure matches (lists of arrays)
            for i in range(len(global_weights['coefs'])):
                self.local_model.coefs_[i] = global_weights['coefs'][i]
            for i in range(len(global_weights['intercepts'])):
                self.local_model.intercepts_[i] = global_weights['intercepts'][i]

    def train_local(self, epochs):
        """Trains the local model on client's data.
        For sklearn models, 'epochs' can be interpreted as the local training effort.
        With warm_start=True, `fit` continues training from current state.
        """
        if self.model_type == 'MLPClassifier':
            # With warm_start=True (set in __init__), subsequent calls to fit continue training.
            # Here, we just call fit, letting it train for its internal max_iter or until convergence
            # based on the model's parameters and `warm_start`.
            self.local_model.fit(self.X_local, self.y_local)

class Server:
    def __init__(self, model_type, feature_count, random_state):
        self.model_type = model_type
        self.global_model = self._initialize_model(model_type, feature_count, random_state)
        print(f"Server initialized with a global {model_type} model.")

    def _initialize_model(self, model_type, feature_count, random_state):
        if model_type == 'MLPClassifier':
            model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500, # Initial max_iter
                random_state=random_state,
                early_stopping=False,
                solver='adam',
                activation='relu',
                warm_start=True
            )
            # Dummy fit to initialize coefs_ and intercepts_
            dummy_X = np.random.rand(2, feature_count)
            dummy_y = np.array([0, 1])
            model.fit(dummy_X, dummy_y)
            return model
        else:
            raise ValueError("Unsupported model type for server.")

    def get_global_weights(self):
        """Returns the current weights/coefficients of the global model."""
        if self.model_type == 'MLPClassifier':
            if hasattr(self.global_model, 'coefs_'):
                return {
                    'coefs': self.global_model.coefs_,
                    'intercepts': self.global_model.intercepts_
                }
        return None # Fallback if weights aren't initialized or model type is unknown

    def set_global_weights(self, new_weights):
        """Sets the global model's weights to the new aggregated weights."""
        if self.model_type == 'MLPClassifier':
            self.global_model.coefs_ = new_weights['coefs']
            self.global_model.intercepts_ = new_weights['intercepts']

    def aggregate_models(self, client_weights_list):
        """Aggregates weights from all clients using Federated Averaging."""
        if not client_weights_list:
            return

        if self.model_type == 'MLPClassifier':
            # MLP has multiple layers, so multiple coefs_ and intercepts_ arrays
            num_layers_coef = len(client_weights_list[0]['coefs'])
            num_layers_intercept = len(client_weights_list[0]['intercepts'])

            avg_coefs = []
            for i in range(num_layers_coef):
                layer_coefs = [w['coefs'][i] for w in client_weights_list]
                avg_coefs.append(np.mean(layer_coefs, axis=0))

            avg_intercepts = []
            for i in range(num_layers_intercept):
                layer_intercepts = [w['intercepts'][i] for w in client_weights_list]
                avg_intercepts.append(np.mean(layer_intercepts, axis=0))

            self.set_global_weights({
                'coefs': avg_coefs, 'intercepts': avg_intercepts
            })

    def evaluate_global_model(self, X_test, y_test):
        """Evaluates the global model on the central test set."""
        y_pred = self.global_model.predict(X_test)
        y_proba = self.global_model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc_val = roc_auc_score(y_test, y_proba)
        auprc = pr_auc_score(y_test, y_proba)
        logloss = log_loss(y_test, y_proba)
        return {
            'Accuracy': acc, 'Precision': prec, 'Recall': rec,
            'F1-score': f1, 'ROC-AUC': roc_auc_val, 'AUPRC': auprc,
            'LogLoss': logloss
        }

print("✅ Client and Server classes defined.")
print("-" * 60)

################################################################################
# Phase 4: Federated Training Loop
################################################################################
print("Phase 4: Starting Federated Training Loop (MLPClassifier) ...")

# Function for AUPRC calculation
def pr_auc_score(y_true, y_probas):
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probas)
    return auc(recall_vals, precision_vals)

# Setup results storage for federated MLP
federated_results_mlp = OrderedDict()

# Store evaluation metrics over rounds for plotting
metrics_history_mlp = {
    'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [],
    'ROC-AUC': [], 'AUPRC': [], 'LogLoss': []
}

if not os.path.exists("fl_plots"):
    os.makedirs("fl_plots")

# --- FL for MLPClassifier ---
print("\n--- Running Federated Learning for MLPClassifier ---")
server_mlp = Server(
    model_type='MLPClassifier',
    feature_count=X_train_resampled_global.shape[1],
    random_state=RANDOM_STATE
)
clients_mlp = [
    Client(
        i + 1, X_train_clients[i], y_train_clients[i],
        'MLPClassifier', RANDOM_STATE
    )
    for i in range(NUM_CLIENTS)
]

for round_num in range(NUM_COMMUNICATION_ROUNDS):
    round_start_time = time.time()
    print(f"\n--- Round {round_num + 1}/{NUM_COMMUNICATION_ROUNDS} "
          f"(MLPClassifier) ---")

    # 1. Server sends global model to clients
    global_weights_mlp = server_mlp.get_global_weights()
    for client in clients_mlp:
        client.set_model_weights(global_weights_mlp)

    # 2. Clients train locally and send updates
    client_weights_list_mlp = []
    for client in clients_mlp:
        client.train_local(CLIENT_LOCAL_EPOCHS)
        client_weights_list_mlp.append(client.get_model_weights())

    # 3. Server aggregates updates
    server_mlp.aggregate_models(client_weights_list_mlp)

    # 4. Server evaluates global model
    eval_metrics_mlp = server_mlp.evaluate_global_model(
        X_test_processed_df, y_test_global
    )
    federated_results_mlp[f'Round {round_num + 1}'] = eval_metrics_mlp

    for metric, value in eval_metrics_mlp.items():
        metrics_history_mlp[metric].append(value)

    print(f"Round {round_num + 1} Metrics: "
          f"Acc={eval_metrics_mlp['Accuracy']:.4f}, "
          f"Prec={eval_metrics_mlp['Precision']:.4f}, "
          f"Rec={eval_metrics_mlp['Recall']:.4f}, "
          f"F1={eval_metrics_mlp['F1-score']:.4f}, "
          f"ROC-AUC={eval_metrics_mlp['ROC-AUC']:.4f}, "
          f"AUPRC={eval_metrics_mlp['AUPRC']:.4f}, "
          f"LogLoss={eval_metrics_mlp['LogLoss']:.4f}")
    print(f"Round completed in {time.time() - round_start_time:.2f} seconds.")


print("\n✅ Federated Training Loop complete.")
print("-" * 60)

################################################################################
# Phase 5: Centralized Model Training (for comparison)
################################################################################
print("Phase 5: Training Centralized MLPClassifier for Comparison...")

# Train a standard (non-federated) MLPClassifier on the combined, preprocessed data
centralized_mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    max_iter=500,
    random_state=RANDOM_STATE,
    early_stopping=False,
    solver='adam',
    activation='relu'
)

start_time_centralized = time.time()
centralized_mlp.fit(X_train_resampled_global, y_train_resampled_global)
print(f"✅ Centralized MLPClassifier trained in "
      f"{time.time() - start_time_centralized:.2f} seconds.")

# Evaluate centralized model
y_pred_centralized = centralized_mlp.predict(X_test_processed_df)
y_proba_centralized = centralized_mlp.predict_proba(X_test_processed_df)[:, 1]

centralized_metrics = {
    'Accuracy': accuracy_score(y_test_global, y_pred_centralized),
    'Precision': precision_score(y_test_global, y_pred_centralized),
    'Recall': recall_score(y_test_global, y_pred_centralized),
    'F1-score': f1_score(y_test_global, y_pred_centralized),
    'ROC-AUC': roc_auc_score(y_test_global, y_proba_centralized),
    'AUPRC': pr_auc_score(y_test_global, y_proba_centralized),
    'LogLoss': log_loss(y_test_global, y_proba_centralized)
}

print("\nCentralized MLPClassifier Performance:")
print(f"Accuracy:  {centralized_metrics['Accuracy']:.4f}")
print(f"Precision: {centralized_metrics['Precision']:.4f}")
print(f"Recall:    {centralized_metrics['Recall']:.4f}")
print(f"F1-Score:  {centralized_metrics['F1-score']:.4f}")
print(f"ROC-AUC:   {centralized_metrics['ROC-AUC']:.4f}")
print(f"AUPRC:     {centralized_metrics['AUPRC']:.4f}")
print(f"LogLoss:   {centralized_metrics['LogLoss']:.4f}")
print("-" * 60)

################################################################################
# Phase 6: Model Evaluation Plots (Limited to 5 images)
################################################################################
print("Phase 6: Generating Model Evaluation Plots (5 total images)...")

# Define function to plot metrics over rounds
def plot_metrics_over_rounds(
    metrics_history, model_name, plot_dir="fl_plots"
):
    num_rounds = len(metrics_history['Accuracy'])
    rounds = range(1, num_rounds + 1)

    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle(f'Federated Learning Metrics Over Rounds ({model_name})',
                 fontsize=16)

    axes = axes.flatten()
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-score',
                       'ROC-AUC', 'AUPRC']

    for i, metric in enumerate(metrics_to_plot):
        axes[i].plot(rounds, metrics_history[metric], marker='o',
                     linestyle='-')
        axes[i].set_title(metric)
        axes[i].set_xlabel("Communication Round")
        axes[i].set_ylabel(metric)
        axes[i].grid(True)
        axes[i].set_xticks(
            np.arange(0, num_rounds + 1, max(1, num_rounds // 10))
        ) # Ensure reasonable ticks

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f"{plot_dir}/fl_metrics_over_rounds_{model_name.lower().replace(' ', '_')}.png")
    plt.close()
    print(f"  - Plot 1/5: FL Metrics Over Rounds for {model_name} saved.")

# Define function for final model plots
def evaluate_and_plot_final_model(
    model, model_name, X_test, y_test, plot_dir="fl_plots", plot_offset=1
):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Confusion Matrix (Plot 2)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["No Heart Disease", "Heart Disease"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name} Global Model - Confusion Matrix")
    plt.savefig(f"{plot_dir}/{model_name.lower().replace(' ', '_')}_final_confusion_matrix.png")
    plt.close()
    print(f"  - Plot {plot_offset+1}/5: Final Confusion Matrix for {model_name} saved.")

    # ROC Curve (Plot 3)
    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(y_test, y_proba, name=model_name, ax=plt.gca())
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.title(f"{model_name} Global Model - ROC Curve")
    plt.legend()
    plt.savefig(f"{plot_dir}/{model_name.lower().replace(' ', '_')}_final_roc_curve.png")
    plt.close()
    print(f"  - Plot {plot_offset+2}/5: Final ROC Curve for {model_name} saved.")

    # Precision-Recall Curve (Plot 4)
    plt.figure(figsize=(8, 6))
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    plt.plot(recall_vals, precision_vals,
             label=f'{model_name} (AUPRC = {pr_auc_score(y_test, y_proba):.4f})')
    no_skill_auprc = len(y_test[y_test==1]) / len(y_test)
    plt.plot([0, 1], [no_skill_auprc, no_skill_auprc], 'k--',
             label=f'No Skill (AUPRC = {no_skill_auprc:.4f})')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{model_name} Global Model - Precision-Recall Curve")
    plt.legend()
    plt.savefig(f"{plot_dir}/{model_name.lower().replace(' ', '_')}_final_pr_curve.png")
    plt.close()
    print(f"  - Plot {plot_offset+3}/5: Final PR Curve for {model_name} saved.")

# 1. Plot FL Metrics Over Rounds (MLPClassifier)
plot_metrics_over_rounds(metrics_history_mlp, "MLPClassifier")

# 2-4. Evaluate and plot final MLPClassifier model
evaluate_and_plot_final_model(server_mlp.global_model, "MLPClassifier",
                              X_test_processed_df, y_test_global)

# 5. Centralized vs. Federated Performance Comparison
print("\n--- Generating Centralized vs. Federated Comparison Plot ---")
best_federated_metrics = pd.DataFrame(federated_results_mlp).T.max().to_dict()
# For LogLoss, we want the minimum, not maximum
best_federated_metrics['LogLoss'] = pd.DataFrame(federated_results_mlp).T['LogLoss'].min()

comparison_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'AUPRC'],
    'Federated MLP': [best_federated_metrics[m] for m in ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'AUPRC']],
    'Centralized MLP': [centralized_metrics[m] for m in ['Accuracy', 'Precision', 'Recall', 'F1-score', 'ROC-AUC', 'AUPRC']]
}
comparison_df = pd.DataFrame(comparison_data)

plt.figure(figsize=(12, 7))
bar_width = 0.35
index = np.arange(len(comparison_df))

plt.bar(index, comparison_df['Federated MLP'], bar_width, label='Federated MLP')
plt.bar(index + bar_width, comparison_df['Centralized MLP'], bar_width, label='Centralized MLP')

plt.xlabel('Metric')
plt.ylabel('Score')
plt.title('Comparison of Centralized vs. Federated MLP Performance')
plt.xticks(index + bar_width / 2, comparison_df['Metric'])
plt.legend()
plt.ylim(0.5, 1.0) # Adjust Y-axis for clarity
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("fl_plots/centralized_vs_federated_mlp_comparison.png")
plt.close()
print("  - Plot 5/5: Centralized vs. Federated Comparison saved.")


print("\n✅ All 5 evaluation plots generated. Check the 'fl_plots' directory.")
print("-" * 60)


################################################################################
# Phase 7: Discussion of Federated Learning Insights (Placeholder for your paper)
################################################################################
print("Phase 7: Analysis and Discussion of Federated Learning Insights (To be done by you)")
print("""
This section is crucial for your research paper. Here are points to consider based on the script's output and the generated plots:

1.  **Convergence of the Federated Model**:
    *   Examine `fl_metrics_over_rounds_mlpclassifier.png`. Discuss how the metrics (Accuracy, F1-score, ROC-AUC, AUPRC) evolve over the communication rounds. Does the model quickly improve and then stabilize? This shows successful learning collaboration without data sharing.

2.  **Performance Analysis of the Federated Model**:
    *   Refer to `mlpclassifier_final_confusion_matrix.png`, `mlpclassifier_final_roc_curve.png`, and `mlpclassifier_final_pr_curve.png`.
    *   **Confusion Matrix**: Discuss True Positives, False Positives, False Negatives, and True Negatives. How well does the model identify actual heart disease cases (Recall) and avoid false alarms (Precision)?
    *   **ROC Curve**: Comment on the model's ability to distinguish between classes. Is the curve close to the top-left corner?
    *   **Precision-Recall Curve**: Given the class imbalance (though SMOTE balanced training, test set reflects original), this curve is particularly important. Discuss its shape and AUPRC score – a higher AUPRC indicates better performance on the minority class.

3.  **Comparison: Federated vs. Centralized Performance**:
    *   Crucially, discuss `centralized_vs_federated_mlp_comparison.png`. How do the "Federated MLP" scores compare to "Centralized MLP" scores?
    *   Is the performance of the federated model comparable to or slightly lower than the centralized one? This illustrates the trade-off: achieving strong performance while preserving data privacy. Highlight that minor performance drops are often acceptable given the significant privacy benefits.

4.  **Benefits and Novelty of Federated Learning**:
    *   Reiterate how FL addresses the core challenge of privacy and data silos in healthcare. Emphasize that your paper demonstrates this on a relevant medical dataset, showcasing the practical applicability of FL in a privacy-sensitive domain.
    *   Mention the novelty of applying FL (especially MLP) to this specific problem, given the limited number of such papers.

5.  **Limitations and Future Work (Concise)**:
    *   Briefly mention that this is a simulation and real-world FL involves more complexities (network latency, client dropouts, more advanced security).
    *   If applicable, discuss data heterogeneity (Non-IID data) as a challenge and briefly suggest FL algorithms designed to address it (e.g., FedProx, FedNova, or personalized FL approaches).
    *   Consider mentioning future directions such as incorporating differential privacy for stronger guarantees or exploring other complex models in a federated setting.
""")
print("-" * 60)
print("Congratulations! Script execution finished.")
