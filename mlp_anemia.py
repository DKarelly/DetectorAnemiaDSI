"""Entrenamiento y evaluación del MLP para clasificar niveles de anemia."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight


CLASS_NAMES = ["sin_anemia", "anemia_leve", "anemia_moderada", "anemia_severa"]
TARGET_COLUMN = "target_anemia_codigo"
EXPECTED_FEATURE_COUNT = 91
RANDOM_STATE = 42


def build_model(input_count: int) -> tf.keras.Model:
    """Construye la arquitectura MLP definida en el informe."""
    regularizer = tf.keras.regularizers.l2(1e-4)
    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(input_count,), name="features"),
            tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=regularizer),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="anemia_level"),
        ],
        name="mlp_anemia",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def class_weights(y: np.ndarray) -> dict[int, float]:
    """Calcula pesos balanceados para las cuatro clases clínicas."""
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_NAMES)),
        y=y,
    )
    return {index: float(weight) for index, weight in enumerate(weights)}


def train_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> tuple[tf.keras.Model, tf.keras.callbacks.History]:
    model = build_model(x_train.shape[1])
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
    )
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=100,
        batch_size=32,
        class_weight=class_weights(y_train),
        callbacks=[early_stopping],
        verbose=0,
    )
    return model, history


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=np.arange(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
            labels=np.arange(len(CLASS_NAMES)),
        ).tolist(),
    }


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(path)
    if TARGET_COLUMN not in frame:
        raise ValueError(f"No existe la columna objetivo requerida: {TARGET_COLUMN}")

    feature_columns = [
        column for column in frame.columns if column not in {"target_anemia_nivel", TARGET_COLUMN}
    ]
    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Se esperaban {EXPECTED_FEATURE_COUNT} features y se encontraron "
            f"{len(feature_columns)}."
        )
    if frame[feature_columns].isna().any().any():
        raise ValueError("Las features contienen valores nulos.")
    if not set(frame[TARGET_COLUMN].unique()).issubset(range(len(CLASS_NAMES))):
        raise ValueError("La variable objetivo contiene códigos fuera de 0, 1, 2 y 3.")

    return (
        frame[feature_columns].to_numpy(dtype=np.float32),
        frame[TARGET_COLUMN].to_numpy(dtype=np.int64),
    )


def run_training(input_path: Path, output_dir: Path) -> dict:
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y = load_dataset(input_path)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cross_validation = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    fold_metrics = []
    for fold, (train_indices, validation_indices) in enumerate(
        cross_validation.split(x_train, y_train),
        start=1,
    ):
        model, history = train_one_fold(
            x_train[train_indices],
            y_train[train_indices],
            x_train[validation_indices],
            y_train[validation_indices],
        )
        predictions = np.argmax(model.predict(x_train[validation_indices], verbose=0), axis=1)
        metrics = evaluate_predictions(y_train[validation_indices], predictions)
        metrics["fold"] = fold
        metrics["epochs"] = len(history.history["loss"])
        fold_metrics.append(metrics)

    final_train_indices, final_validation_indices = train_test_split(
        np.arange(len(y_train)),
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
    final_model, final_history = train_one_fold(
        x_train[final_train_indices],
        y_train[final_train_indices],
        x_train[final_validation_indices],
        y_train[final_validation_indices],
    )
    test_predictions = np.argmax(final_model.predict(x_test, verbose=0), axis=1)
    test_metrics = evaluate_predictions(y_test, test_predictions)
    test_metrics["epochs"] = len(final_history.history["loss"])

    final_model.save(output_dir / "mlp_anemia.keras")
    pd.DataFrame(
        test_metrics["confusion_matrix"],
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(output_dir / "matriz_confusion.csv", encoding="utf-8-sig")

    results = {
        "dataset": str(input_path),
        "random_state": RANDOM_STATE,
        "feature_count": int(x.shape[1]),
        "train_records": int(len(y_train)),
        "test_records": int(len(y_test)),
        "class_weights": class_weights(y_train),
        "cross_validation_4_folds": fold_metrics,
        "test_metrics": test_metrics,
    }
    (output_dir / "metricas.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ENDES_2025_MLP_macrorregion_norte.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    results = run_training(args.input, args.output_dir)
    metrics = results["test_metrics"]
    print(f"Accuracy de prueba: {metrics['accuracy']:.4f}")
    print(f"F1 macro de prueba: {metrics['f1_macro']:.4f}")
    print(f"Resultados guardados en: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
