from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier


def build_model(name: str, vectorizer_kind: str = "binary"):
    """
    Build the six classifiers used in the reference paper.

    The paper reports:
      Logistic Regression: L2, tolerance 1e-4
      SVM: L2, tolerance 1e-4
      Multinomial NB: Laplace smoothing 1
      Random Forest: 100 trees, Gini
      Boosting: 100 trees, learning rate 0.1
      DNN: hidden layers (30,30,20,10,10), logistic activation,
           L2=0.0001, early stopping, Adam
    """
    if name == "logistic":
        return LogisticRegression(
            C=1.0,
            penalty="l2",
            tol=1e-4,
            solver="liblinear",
            max_iter=2000,
        )

    if name == "svm":
        # The paper's result table reports C=100 for binary,
        # C=20 for word-count, and C=1 for TF-IDF.
        c_by_vectorizer = {"binary": 100.0, "count": 20.0, "tfidf": 1.0}
        return SVC(
            C=c_by_vectorizer.get(vectorizer_kind, 1.0),
            kernel="linear",
            probability=True,
            tol=1e-4,
        )

    if name == "naive_bayes":
        return MultinomialNB(alpha=1.0)

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1,
            random_state=42,
        )

    if name == "boosting":
        return GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            min_samples_split=2,
            random_state=42,
        )

    if name == "dnn":
        return MLPClassifier(
            hidden_layer_sizes=(30, 30, 20, 10, 10),
            activation="logistic",
            alpha=0.0001,
            early_stopping=True,
            solver="adam",
            max_iter=100,
            random_state=42,
        )

    raise ValueError(f"Unknown model: {name}")
