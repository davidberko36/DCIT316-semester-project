import os
import argparse
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType
import warnings
warnings.filterwarnings("ignore")


def load_tsv(path, label_col=0, text_col=1):
    df = pd.read_csv(path, sep='\t', header=None, quoting=3, dtype = str, engine='python')
    df = df.fillna('')
    labels = df.iloc[:, label_col].astype(str).tolist()
    texts = df.iloc[:, text_col].astype(str).tolist()
    return texts, labels


def main(args):
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)
    print("Loading training data:", args.train)
    X_train, y_train = load_tsv(args.train, label_col=args.label_col, text_col=args.text_col)
    
    X_valid, y_valid = (None, None)
    if args.valid and os.path.exists(args.valid):
        print("Loading valid data:", args.valid)
        X_valid, y_valid = load_tsv(args.valid, label_col=args.label_col, text_col=args.text_col)
        
        print("Building pipeline...")
        pipe = Pipeline([
            ("tfidif", TfidfVectorizer(ngram_range=(1,2), max_features=50000)),
            ("clf", LogisticRegression(max_iter=2000, solver='lbfgs', class_weight='balanced'))
        ])

        print("Fitting pipeline on training data...")
        pipe.fit(X_train, y_train)

        if X_valid:
            print("Evaluating on validation set...")
            preds = pipe.predict(X_valid)
            print("Accuracy:", accuracy_score(y_valid, preds))
            print(classification_report(y_valid, preds))

        joblib_path = args.out_model.replace('.onnx', '.joblib')
        print("Saving sklearn pipeline to", joblib_path)
        joblib.dump(pipe, joblib_path)

        print("Exporting pipeline to ONNX:", args.out_model)
        initial_type = [("input", StringTensorType([None, 1]))]
        onnx_model = convert_sklearn(pipe, initial_types=initial_type, target_opset=13)
        
        try:
            onnx_model = convert_sklearn(pipe, initial_types=initial_type)
        except Exception as e:
            print("ONNX conversion failed:", e)
            raise

        if isinstance(onnx_model, tuple):
            model_proto = onnx_model[0]
        else:
            model_proto = onnx_model

        with open(args.out_model, "wb") as f:
            f.write(model_proto.SerializeToString())
        print("Done. ONNX saved to", args.out_model)
        print("Validate with: python -c\"import onnx; onnx.checker.check_model(onnx.load('{}'))\"",format(args.out_model))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train LR on LIAR dataset and export to ONNX")
    p.add_argument("--train", default="Data/liar_dataset/train.csv")
    p.add_argument("--valid", default="Data/liar_dataset/valid.tsv")
    p.add_argument("--out-model", default="models/lr_pipeline.onnx")
    p.add_argument("--label-col", type=int, default=0, help="index of label column in TSV (o-based)")
    p.add_argument("--text-col", type=int, default=1, help="index of text/statement column in TSV (0-based)")
    args = p.parse_args()
    main(args)
