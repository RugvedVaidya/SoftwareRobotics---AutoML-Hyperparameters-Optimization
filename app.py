import os
import datetime
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from werkzeug.utils import secure_filename
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO
import json

# ✅ Initialize Flask App
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MODEL_FOLDER"] = "models"
app.config["SCALER_FOLDER"] = "scalers"
app.config["GRAPH_FOLDER"] = "graphs"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["MODEL_FOLDER"], exist_ok=True)
os.makedirs(app.config["SCALER_FOLDER"], exist_ok=True)
os.makedirs(app.config["GRAPH_FOLDER"], exist_ok=True)


# ✅ Home Route
@app.route("/", methods=["GET"])
def index():
    available_models = {
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "lightgbm": "LightGBM",
        "logistic_regression": "Logistic Regression",
        "knn": "KNN"
    }
    return render_template("index.html", available_models=available_models)


# ✅ Upload & Process Route
@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files["file"]
        target_column = request.form.get("target_column", "").strip()
        selected_models = request.form.getlist("selected_models")

        if file.filename == "":
            return jsonify({"error": "No selected file"}), 400
        if not target_column:
            return jsonify({"error": "Target column is required"}), 400
        if not selected_models:
            return jsonify({"error": "At least one model must be selected"}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(file_path)

        df = pd.read_csv(file_path)
        X, y, scaler, is_classification = clean_and_preprocess_data(df, target_column)

        model_name_mapping = {
            "random_forest": "Random Forest",
            "xgboost": "XGBoost",
            "lightgbm": "LightGBM",
            "logistic_regression": "Logistic Regression",
            "knn": "KNN"
        }
        selected_model_names = [model_name_mapping[model] for model in selected_models if model in model_name_mapping]

        results, best_model_name, model_path, all_roc_auc_plots, all_confusion_matrix_plots, all_pr_curve_plots = train_and_compare_models(X, y, is_classification, selected_model_names)
        hyperparameters = suggest_hyperparameters(selected_model_names)

        return render_template(
            "result.html",
            results=results,
            best_model=best_model_name,
            model_path=model_path,
            scaler_path="N/A",
            model_type="classification" if is_classification else "regression",
            hyperparameters=hyperparameters,
            selected_models=selected_model_names,
            roc_auc_plots=all_roc_auc_plots if is_classification else None,
            confusion_matrix_plots=all_confusion_matrix_plots if is_classification else None,
            pr_curve_plots=all_pr_curve_plots if is_classification else None
        )

    except Exception as e:
        return jsonify({"error": f"Error processing dataset: {str(e)}"}), 500


# ✅ Data Preprocessing
def clean_and_preprocess_data(df, target_column):
    df = df.copy()
    df.columns = df.columns.str.replace(r"[^a-zA-Z0-9_]", "_", regex=True).str.strip("_")
    df.drop_duplicates(inplace=True)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "MISSING", inplace=True)
        else:
            df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0, inplace=True)
    drop_cols = [col for col in df.columns if 'id' in col.lower() or 'timestamp' in col.lower()]
    df.drop(columns=drop_cols, errors='ignore', inplace=True)
    X = df.drop(columns=[target_column])
    y = df[target_column]
    is_classification = len(y.unique()) < 10
    if is_classification and y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)
    categorical_cols = X.select_dtypes(include=['object']).columns
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    scaler = StandardScaler()
    X.iloc[:, :] = scaler.fit_transform(X)
    return X, y, scaler, is_classification


# ✅ Model Training & Selection
def train_and_compare_models(X, y, is_classification, selected_models=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if is_classification and len(np.unique(y_train)) > 1:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
    scoring = 'f1_weighted' if is_classification else 'neg_mean_squared_error'
    param_grids = {
        "Random Forest": {'model': RandomForestClassifier(random_state=42), 'params': {'n_estimators': [100, 200], 'max_depth': [None, 10]}},
        "XGBoost": {'model': XGBClassifier(eval_metric="mlogloss" if is_classification else "rmse", random_state=42), 'params': {'n_estimators': [100], 'learning_rate': [0.01, 0.1]}},
        "LightGBM": {'model': LGBMClassifier(random_state=42), 'params': {'n_estimators': [100], 'learning_rate': [0.01, 0.1]}},
        "Logistic Regression": {'model': LogisticRegression(max_iter=1000, random_state=42), 'params': {'C': [0.1, 1.0], 'solver': ['liblinear']}},
        "KNN": {'model': KNeighborsClassifier(), 'params': {'n_neighbors': [3, 5], 'weights': ['uniform', 'distance']}}
    }
    if selected_models:
        param_grids = {k: v for k, v in param_grids.items() if k in selected_models}

    best_score = -np.inf
    best_model = None
    best_model_name = ""
    all_roc_auc_plots = {}
    all_confusion_matrix_plots = {}
    all_pr_curve_plots = {}
    results = {}

    for name, model_info in param_grids.items():
        model = model_info['model']
        param_grid = model_info['params']
        search = RandomizedSearchCV(model, param_grid, n_iter=5, cv=3, scoring=scoring, n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)
        final_model = search.best_estimator_
        y_pred = final_model.predict(X_test)

        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            results[name] = {'accuracy': round(accuracy * 100, 2), 'f1_score': round(f1 * 100, 2), 'best_params': search.best_params_}
            score = f1

            # ROC Curve
            if len(np.unique(y_test)) > 2:
                y_pred_proba = final_model.predict_proba(X_test)
                fpr = {}; tpr = {}; roc_auc = {}
                for i in range(len(np.unique(y_test))):
                    fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_proba[:, i]); roc_auc[i] = auc(fpr[i], tpr[i])
                plt.figure(figsize=(8, 6));
                for i in range(len(np.unique(y_test))): plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
                plt.plot([0, 1], [0, 1], 'k--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05]); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title(f'ROC Curve - {name}'); plt.legend(loc="lower right"); img = BytesIO(); plt.savefig(img, format='png'); img.seek(0); plot_url = base64.b64encode(img.getvalue()).decode('utf8'); all_roc_auc_plots[name] = plot_url; plt.close()
            elif len(np.unique(y_test)) == 2:
                y_pred_proba = final_model.predict_proba(X_test)[:, 1]; fpr, tpr, _ = roc_curve(y_test, y_pred_proba); roc_auc_score = auc(fpr, tpr); plt.figure(figsize=(8, 6)); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_score:.2f})'); plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05]); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title(f'ROC Curve - {name}'); plt.legend(loc="lower right"); img = BytesIO(); plt.savefig(img, format='png'); img.seek(0); plot_url = base64.b64encode(img.getvalue()).decode('utf8'); all_roc_auc_plots[name] = plot_url; plt.close()

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred); plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y)); plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title(f'Confusion Matrix - {name}'); img = BytesIO(); plt.savefig(img, format='png'); img.seek(0); plot_url = base64.b64encode(img.getvalue()).decode('utf8'); all_confusion_matrix_plots[name] = plot_url; plt.close()

            # Precision-Recall Curve
            if len(np.unique(y_test)) == 2:
                y_pred_proba = final_model.predict_proba(X_test)[:, 1]; precision, recall, _ = precision_recall_curve(y_test, y_pred_proba); avg_precision = average_precision_score(y_test, y_pred_proba); plt.figure(figsize=(8, 6)); plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.2f})'); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'Precision-Recall Curve - {name}'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05]); plt.legend(loc="lower left"); plt.grid(True); img = BytesIO(); plt.savefig(img, format='png'); img.seek(0); plot_url = base64.b64encode(img.getvalue()).decode('utf8'); all_pr_curve_plots[name] = plot_url; plt.close()
            elif len(np.unique(y_test)) > 2:
                y_pred_proba = final_model.predict_proba(X_test); precision = {}; recall = {}; avg_precision = {}
                plt.figure(figsize=(8, 6));
                for i in range(len(np.unique(y_test))):
                    precision[i], recall[i], _ = precision_recall_curve(y_test == i, y_pred_proba[:, i]); avg_precision[i] = average_precision_score(y_test == i, y_pred_proba[:, i]); plt.plot(recall[i], precision[i], lw=2, label=f'Class {i} (AP = {avg_precision[i]:.2f})')
                plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'Precision-Recall Curve - {name}'); plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05]); plt.legend(loc="lower left"); plt.grid(True); img = BytesIO(); plt.savefig(img, format='png'); img.seek(0); plot_url = base64.b64encode(img.getvalue()).decode('utf8'); all_pr_curve_plots[name] = plot_url; plt.close()

        else:
            mse = mean_squared_error(y_test, y_pred); rmse = np.sqrt(mse); results[name] = {'rmse': round(rmse, 4), 'best_params': search.best_params_}; score = -rmse

        if score > best_score:
            best_score = score; best_model = final_model; best_model_name = name

    if not results:
        return {}, "No models selected", None, {}, {}, {}

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(app.config["MODEL_FOLDER"], f"{best_model_name}_{timestamp}.pkl")
    joblib.dump(best_model, model_path)

    return results, best_model_name, model_path, all_roc_auc_plots, all_confusion_matrix_plots, all_pr_curve_plots


# ✅ Suggest Hyperparameters (only for selected models)
def suggest_hyperparameters(selected_models=None):
    all_hyperparameters = {
        "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 10]},
        "XGBoost": {"n_estimators": [100], "learning_rate": [0.01, 0.1]},
        "LightGBM": {"n_estimators": [100], "learning_rate": [0.01, 0.1]},
        "Logistic Regression": {"C": [0.1, 1.0], "solver": ["liblinear"]},
        "KNN": {"n_neighbors": [3, 5], "weights": ["uniform", "distance"]},
    }
    if selected_models:
        return {k : v for k, v in all_hyperparameters.items() if k in selected_models}
    return all_hyperparameters


# ✅ Download Report Route
@app.route("/download_report")
def download_report():
    report_data = {
        "best_model": request.args.get("best_model"),
        "model_type": request.args.get("model_type"),
        "results": json.loads(request.args.get("results")),
        "hyperparameters": json.loads(request.args.get("hyperparameters"))
    }

    report_filename = "model_training_report.json"
    report_path = os.path.join(app.config["UPLOAD_FOLDER"], report_filename)

    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=4)

    return send_file(report_path, as_attachment=True, download_name=report_filename)


# ✅ Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
