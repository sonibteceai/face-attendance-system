import numpy as np
import pickle
import os
from collections import Counter

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
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

    # ---- KNN with cosine distance only (lightweight: no multi-model comparison) ----
    pipeline = Pipeline([("clf", KNeighborsClassifier(metric="cosine"))])
    params = {
        "clf__n_neighbors": [1, 3, 5, 7],
        "clf__weights": ["uniform", "distance"],
    }

    search = GridSearchCV(
        pipeline, params,
        cv=cv, scoring="accuracy", n_jobs=1
    )
    search.fit(X, y)

    best_overall = search.best_estimator_
    best_score = search.best_score_
    best_name = "KNN_cosine"

    print(f"{best_name:20s} best CV accuracy: {best_score:.4f}  params: {search.best_params_}")
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