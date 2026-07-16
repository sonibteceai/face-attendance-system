import numpy as np
import pickle
import os
from collections import Counter

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from db import get_connection

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "classifier.pkl")


def load_embeddings():
    conn = get_connection()
    conn.sync()  # pull latest data from Turso before reading
    cursor = conn.cursor()

    cursor.execute("SELECT student_id, embedding FROM face_embeddings")
    emb_rows = cursor.fetchall()

    cursor.execute("SELECT student_id, name FROM student_profiles")
    names = dict(cursor.fetchall())
    conn.close()

    X, y = [], []
    for student_id, embedding_blob in emb_rows:
        X.append(np.frombuffer(embedding_blob, dtype=np.float32))
        y.append(student_id)

    return np.array(X), np.array(y), names


def get_cv_folds(y):
    """Safely pick number of CV folds based on smallest class size."""
    smallest_class = min(Counter(y).values())
    return max(2, min(3, smallest_class))


def save_model_to_db(model, names, model_type):
    """Persist the trained classifier in Turso so it survives app restarts
    on ephemeral hosts like Streamlit Community Cloud."""
    model_bytes = pickle.dumps({"model": model, "names": names, "model_type": model_type})

    conn = get_connection()
    cursor = conn.cursor()
    # Single-row table: always id=1, replace on retrain.
    cursor.execute("DELETE FROM model_store WHERE id = 1")
    cursor.execute(
        "INSERT INTO model_store (id, model_blob, model_type, updated_at) VALUES (1, ?, ?, CURRENT_TIMESTAMP)",
        (model_bytes, model_type),
    )
    conn.commit()
    conn.sync()
    conn.close()


def load_model_from_db():
    """Load the trained classifier from Turso. Returns dict with
    'model', 'names', 'model_type', or None if nothing has been trained yet."""
    conn = get_connection()
    conn.sync()  # pull latest data before reading
    cursor = conn.cursor()
    cursor.execute("SELECT model_blob FROM model_store WHERE id = 1")
    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None
    return pickle.loads(row[0])


def train_and_save():
    X, y, names = load_embeddings()

    if len(X) == 0:
        print("❌ No students registered yet.")
        return False

    n_classes = len(set(y))
    if n_classes < 2:
        print("❌ Need at least 2 different students to train a classifier.")
        return False

    cv_folds = get_cv_folds(y)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    print(f"Training on {len(X)} samples, {n_classes} students, using {cv_folds}-fold CV\n")

    # ---- Define candidate models + their search grids ----
    candidates = {
        "KNN": {
            "pipeline": Pipeline([("clf", KNeighborsClassifier())]),
            "params": {
                "clf__n_neighbors": [1, 3, 5],
                "clf__metric": ["cosine", "euclidean"],
                "clf__weights": ["uniform", "distance"],
            },
        },
        "SVM_Linear": {
            "pipeline": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="linear", probability=True))]),
            "params": {
                "clf__C": [0.1, 1, 10],
            },
        },
        "SVM_RBF": {
            "pipeline": Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", probability=True))]),
            "params": {
                "clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale", "auto"],
            },
        },
        "LogisticRegression": {
            "pipeline": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))]),
            "params": {
                "clf__C": [0.1, 1, 10],
            },
        },
        "RandomForest": {
            "pipeline": Pipeline([("clf", RandomForestClassifier(random_state=42))]),
            "params": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 10],
            },
        },
    }

    best_overall = None
    best_score = -1
    best_name = None
    results = []

    for name, cfg in candidates.items():
        try:
            search = GridSearchCV(
                cfg["pipeline"], cfg["params"],
                cv=cv, scoring="accuracy", n_jobs=-1
            )
            search.fit(X, y)
            results.append((name, search.best_score_, search.best_params_))
            print(f"{name:20s} best CV accuracy: {search.best_score_:.4f}  params: {search.best_params_}")

            if search.best_score_ > best_score:
                best_score = search.best_score_
                best_overall = search.best_estimator_
                best_name = name

        except Exception as e:
            print(f"{name:20s} skipped due to error: {e}")

    print(f"\n🏆 Best model: {best_name} (CV accuracy: {best_score:.4f})")

    # ---- Save best model: Turso (persistent, source of truth) ----
    save_model_to_db(best_overall, names, best_name)
    print(f"✅ Saved best classifier ({best_name}) to Turso model_store")

    # ---- Also cache a local copy for this running session (best-effort) ----
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump({"model": best_overall, "names": names, "model_type": best_name}, f)
        print(f"✅ Cached classifier locally at {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Could not write local cache (non-fatal): {e}")

    return True


if __name__ == "__main__":
    train_and_save()