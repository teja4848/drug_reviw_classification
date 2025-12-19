import joblib
import numpy as np
from pathlib import Path

VECTORIZER_PATH = "models/tfidf_vectorizer.joblib"

EFFECTIVENESS_ORDER = [
    "Ineffective",
    "Marginally Effective",
    "Moderately Effective",
    "Considerably Effective",
    "Highly Effective",
]

SAMPLES = [
    "This medicine worked very well. My symptoms improved in 2 days and no side effects.",
    "It helped a little but I still had pain and some nausea.",
    "Did not work at all. Felt worse and had strong headache and dizziness.",
]

def test_model(model_path: str):
    print("\n" + "=" * 70)
    print(f"Testing model: {model_path}")
    print("=" * 70)

    try:
        model = joblib.load(model_path)
        tfidf = joblib.load(VECTORIZER_PATH)

        X = tfidf.transform(SAMPLES)  # ✅ 2D numeric sparse matrix

        preds = model.predict(X)

        print("Predictions:")
        for i, p in enumerate(preds, 1):
            p_int = int(p)
            label = EFFECTIVENESS_ORDER[p_int] if 0 <= p_int < len(EFFECTIVENESS_ORDER) else str(p)
            print(f"  Sample {i}: {label}")

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            print("\nTop-3 probabilities:")
            for i in range(proba.shape[0]):
                top = np.argsort(proba[i])[::-1][:3]
                top3 = [(EFFECTIVENESS_ORDER[j], float(proba[i][j])) for j in top]
                print(f"  Sample {i+1}: {top3}")

        print(f"\n✅ {Path(model_path).name} - SUCCESS")
        return True

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

def main():
    models = [
        "models/global_best_model.pkl",
        "models/global_best_model_optuna.pkl",
    ]

    print("Testing drug review classification models...")
    results = [test_model(m) for m in models]

    print("\n" + "=" * 70)
    print(f"SUMMARY: {sum(results)}/{len(results)} models passed")
    print("=" * 70)

    raise SystemExit(0 if all(results) else 1)

if __name__ == "__main__":
    main()
