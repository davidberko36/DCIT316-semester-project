import os
import sys
import time
import argparse
import json
import numpy as np
import pandas as pd
import onnxruntime as ort
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from tqdm import tqdm

# Import from existing scripts
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.logistic_regression import load_tsv

# Add missing functions
def encode_text_svd(texts, max_features=20000, n_components=128):
    """
    Encodes text using TF-IDF and dimensionality reduction with SVD
    
    Args:
        texts: List of text strings to encode
        max_features: Maximum number of features for TF-IDF
        n_components: Dimensionality of the output embeddings
        
    Returns:
        Numpy array of shape (len(texts), n_components)
    """
    # Ensure we don't use more features than we have texts
    actual_max_features = min(max_features, len(texts) * 10)
    actual_n_components = min(n_components, int(len(texts) * 0.8))
    
    # Create and fit TF-IDF vectorizer
    tfidf = TfidfVectorizer(max_features=actual_max_features, ngram_range=(1, 2))
    X = tfidf.fit_transform(texts)  # sparse matrix
    
    # Apply SVD for dimensionality reduction
    svd = TruncatedSVD(n_components=actual_n_components, random_state=42)
    X_svd = svd.fit_transform(X)
    
    # Normalize the vectors
    X_norm = normalize(X_svd, norm='l2')
    
    return X_norm

def evaluate_fake_news_model(model_path, test_data_path, label_col=1, text_col=2, output_path=None):
    """
    Evaluate the fake news detection model (logistic regression) on test data.
    
    Args:
        model_path: Path to the ONNX model
        test_data_path: Path to the test data in TSV format
        label_col: Column index for the label in the TSV
        text_col: Column index for the text in the TSV
        output_path: Optional path to save results
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING FAKE NEWS DETECTION MODEL")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Load test data
    print(f"Loading test data from {test_data_path}")
    X_test, y_true = load_tsv(test_data_path, label_col=label_col, text_col=text_col)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return None
    
    # Load ONNX model
    print(f"Loading ONNX model from {model_path}")
    sess = ort.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    
    # Perform predictions
    print("Running inference on test data...")
    y_pred = []
    y_prob = []
    
    batch_size = 32
    for i in tqdm(range(0, len(X_test), batch_size)):
        batch = X_test[i:i+batch_size]
        # ONNX model expects string tensor input
        inputs = np.array(batch).reshape(-1, 1).astype(str)
        pred_onx = sess.run([label_name], {input_name: inputs})[0]
        
        # Convert prediction to list for extending
        if isinstance(pred_onx, np.ndarray):
            y_pred.extend(pred_onx.tolist())
        elif hasattr(pred_onx, '__iter__'):
            try:
                y_pred.extend(list(pred_onx))
            except:
                print(f"Warning: Could not convert predictions to list. Type: {type(pred_onx)}")
                y_pred.extend([str(pred_onx)] * len(batch))
        else:
            print(f"Warning: Unexpected prediction type: {type(pred_onx)}")
            y_pred.extend([str(pred_onx)] * len(batch))
        
        # Try to get probability scores if available
        try:
            prob_name = sess.get_outputs()[1].name
            prob_onx = sess.run([prob_name], {input_name: inputs})[0]
            if isinstance(prob_onx, np.ndarray) and len(prob_onx.shape) > 1:
                # For binary classification, take the second column (positive class)
                y_prob.extend(prob_onx[:, 1].tolist())
        except Exception as e:
            print(f"Note: Could not get probability scores: {e}")
    
    # Convert predictions to same type as true values
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Get unique classes to handle potential label mismatches
    classes = sorted(list(set(y_true) | set(y_pred)))
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0, labels=classes)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0, labels=classes)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=classes)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    # Detailed classification report
    report = classification_report(y_true, y_pred, zero_division=0, labels=classes, output_dict=True)
    
    # Calculate AUC if probabilities available
    auc = None
    if len(y_prob) > 0:
        try:
            # Check if binary classification or convert to binary
            if len(set(y_true)) == 2:
                binary_y_true = np.array([1 if y == classes[-1] else 0 for y in y_true])
                auc = roc_auc_score(binary_y_true, y_prob)
        except:
            pass
    
    # Collect results
    evaluation_time = time.time() - start_time
    results = {
        "model_type": "Fake News Detection (Logistic Regression)",
        "model_path": model_path,
        "test_data": test_data_path,
        "samples_count": len(X_test),
        "classes": classes,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "evaluation_time_seconds": evaluation_time
    }
    
    # Print results
    print(f"\nFake News Detection Model Evaluation Results:")
    print(f"Number of test samples: {len(X_test)}")
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (weighted): {precision:.4f}")
    print(f"Recall (weighted): {recall:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    if auc:
        print(f"AUC: {auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0, labels=classes))
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")
    
    # Save results if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return results

# Import this from neural_nets.py for compatibility
class MINDBehaviorDataset(torch.utils.data.Dataset):
    """
    Reads behaviors.tsv lines:
    userId \t time \t history(space-separated news ids) \t impressions(space-separated newsid-0/1)
    Produces items: (history_ids: List[str], cand_ids: List[str], labels: List[int])
    """
    
    def __init__(self, behaviors_path, news_in_vocab, max_history=50, neg_sample_size=4, sample_negatives=True):
        self.samples = []
        self.max_history = max_history
        self.neg_sample_size = neg_sample_size
        self.sample_negatives = sample_negatives
        
        total_lines = 0
        skipped_lines = 0
        
        # Try pandas first - more robust to various file formats
        try:
            df = pd.read_csv(behaviors_path, sep='\t', header=None, quoting=3, dtype=str, engine='python').fillna('')
            print(f"Loaded behaviors file with shape: {df.shape}")
            
            # Determine column indices
            history_col = 3  # Default for MIND
            impressions_col = 4  # Default for MIND
            
            if df.shape[1] >= 5:
                # Standard MIND format
                print(f"Using columns: history={history_col}, impressions={impressions_col}")
            else:
                # Try to infer column positions
                history_col = df.shape[1] - 2
                impressions_col = df.shape[1] - 1
                print(f"Using inferred columns: history={history_col}, impressions={impressions_col}")
            
            total_lines = len(df)
            
            for _, row in df.iterrows():
                # Extract history
                history = row[history_col] if history_col < len(row) else ""
                history_ids = [h.strip() for h in str(history).split() if h.strip() and h.strip() in news_in_vocab]
                history_ids = history_ids[-max_history:] if history_ids else []
                
                # Extract impressions
                impressions = row[impressions_col] if impressions_col < len(row) else ""
                
                cand_ids = []
                labels = []
                
                # Process impressions (try different formats)
                for imp in str(impressions).split():
                    if '-' in imp:
                        try:
                            nid, lab = imp.rsplit('-', 1)
                            nid = nid.strip()
                            if nid not in news_in_vocab:
                                continue
                            lab_int = int(lab)
                            cand_ids.append(nid)
                            labels.append(lab_int)
                        except Exception:
                            continue
                
                # If no candidates found, try alternative formats
                if not cand_ids and impressions:
                    # Try comma-separated format
                    for imp in str(impressions).split(','):
                        if '-' in imp:
                            try:
                                nid, lab = imp.rsplit('-', 1)
                                nid = nid.strip()
                                if nid not in news_in_vocab:
                                    continue
                                lab_int = int(lab)
                                cand_ids.append(nid)
                                labels.append(lab_int)
                            except Exception:
                                continue
                
                if cand_ids:
                    self.samples.append((history_ids, cand_ids, labels))
                else:
                    skipped_lines += 1
        
        except Exception as e:
            print(f"Pandas loading failed: {e}. Falling back to manual line parsing.")
            
            # Fall back to manual line parsing
            with open(behaviors_path, 'r', encoding='utf-8') as fh:
                for line in fh:
                    total_lines += 1
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) < 3:
                        skipped_lines += 1
                        continue
                    
                    # Extract history and impressions - handle different formats
                    if len(parts) >= 5:  # Standard MIND format
                        history = parts[3]
                        impressions = parts[4]
                    else:
                        # Try to be robust with column positions
                        history = parts[-2] if len(parts) >= 3 else ""
                        impressions = parts[-1]
                    
                    history_ids = [h.strip() for h in history.split() if h.strip() and h.strip() in news_in_vocab]
                    
                    cand_ids = []
                    labels = []
                    
                    # Process impressions (try different formats)
                    for imp in impressions.split():
                        if '-' in imp:
                            try:
                                nid, lab = imp.rsplit('-', 1)
                                nid = nid.strip()
                                if nid not in news_in_vocab:
                                    continue
                                lab_int = int(lab)
                                cand_ids.append(nid)
                                labels.append(lab_int)
                            except Exception:
                                continue
                    
                    if cand_ids:
                        self.samples.append((history_ids[-max_history:], cand_ids, labels))
                    else:
                        skipped_lines += 1
        
        print(f"Loaded {len(self.samples)} valid samples from {behaviors_path}")
        print(f"Total lines: {total_lines}, Skipped: {skipped_lines}")
        print(f"Retention rate: {100 * (total_lines - skipped_lines) / max(1, total_lines):.2f}%")
        
        if self.samples:
            print("First sample:")
            h, c, l = self.samples[0]
            print(f"  History: {h[:5]}...")
            print(f"  Candidates: {c[:5]}...")
            print(f"  Labels: {l[:5]}...")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        history_ids, cand_ids, labels = self.samples[idx]
        history_ids = history_ids[-self.max_history:]
        
        # Optionally sample negatives to reduce candidate set size
        if self.sample_negatives:
            pos_idx = [i for i, l in enumerate(labels) if l == 1]
            neg_idx = [i for i, l in enumerate(labels) if l == 0]
            
            if len(pos_idx) == 0:
                if len(neg_idx) == 0:
                    return history_ids, cand_ids, labels
                chosen = np.random.choice(neg_idx, min(self.neg_sample_size, len(neg_idx)), replace=False).tolist()
            else:
                chosen = list(pos_idx)
                if neg_idx:
                    chosen.extend(np.random.choice(neg_idx, min(self.neg_sample_size, len(neg_idx)), replace=False).tolist())
            
            # Keep order and remove duplicates
            seen = set()
            final_idx = []
            for i in chosen:
                if i not in seen:
                    seen.add(i)
                    final_idx.append(i)
            
            return history_ids, [cand_ids[i] for i in final_idx], [labels[i] for i in final_idx]
        else:
            return history_ids, cand_ids, labels

class ScoringModel(torch.nn.Module):
    def __init__(self, emb_dim, hidden_dim=256):
        super().__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)
        )

    def forward(self, user_emb, cand_emb):
        # user_emb: (B, D)
        # cand_emb: (B, C, D)
        B, C, D = cand_emb.shape
        user_expand = user_emb.unsqueeze(1).expand(-1, C, -1)  # (B,C,D)
        x = torch.cat([user_expand, cand_emb], dim=-1).view(B * C, 2 * D)  # (B*C, 2D)
        logits = self.mlp(x).view(B, C)  # (B, C)
        return logits

def collate_batch(batch, newsid_to_idx, news_emb_np, device):
    """
    Custom collate function for batching samples
    """
    if not batch:
        D = news_emb_np.shape[1]
        return (torch.zeros((0, D), device=device, dtype=torch.float32),
                torch.zeros((0, 0, D), device=device, dtype=torch.float32),
                torch.zeros((0, 0), device=device, dtype=torch.float32))
    
    user_embs = []
    cand_embs = []
    labels = []
    
    for history_ids, cand_ids, lbls in batch:
        # Calculate user embedding (average of history news embeddings)
        history_indices = [newsid_to_idx[nid] for nid in history_ids if nid in newsid_to_idx]
        if not history_indices:
            continue  # Skip if no valid history items
        
        user_emb = news_emb_np[history_indices].mean(axis=0)
        
        # Get candidate embeddings
        valid_indices = []
        valid_labels = []
        for i, nid in enumerate(cand_ids):
            if nid in newsid_to_idx:
                valid_indices.append(newsid_to_idx[nid])
                valid_labels.append(lbls[i])
        
        if not valid_indices:
            continue  # Skip if no valid candidates
            
        cand_emb = news_emb_np[valid_indices]
        
        user_embs.append(user_emb)
        cand_embs.append(cand_emb)
        labels.append(valid_labels)
    
    if not user_embs:
        D = news_emb_np.shape[1]
        return (torch.zeros((0, D), device=device, dtype=torch.float32),
                torch.zeros((0, 0, D), device=device, dtype=torch.float32),
                torch.zeros((0, 0), device=device, dtype=torch.float32))
    
    # Convert to tensors
    user_embs = torch.tensor(np.stack(user_embs), device=device, dtype=torch.float32)
    
    # Handle variable-length candidate lists by padding
    D = news_emb_np.shape[1]
    max_candidates = max(c.shape[0] for c in cand_embs)
    
    # Pad candidate embeddings and labels
    padded_cand_embs = []
    padded_labels = []
    
    for i, (c_emb, l) in enumerate(zip(cand_embs, labels)):
        # Pad or truncate candidate embeddings
        if c_emb.shape[0] > max_candidates:
            # Truncate
            padded_cand_embs.append(c_emb[:max_candidates])
            padded_labels.append(l[:max_candidates])
        elif c_emb.shape[0] < max_candidates:
            # Pad with zeros
            pad_shape = ((0, max_candidates - c_emb.shape[0]), (0, 0))
            padded_emb = np.pad(c_emb, pad_shape, mode='constant')
            padded_cand_embs.append(padded_emb)
            
            # Pad labels with zeros (non-relevant)
            padded_l = l + [0] * (max_candidates - len(l))
            padded_labels.append(padded_l)
        else:
            # Already the right size
            padded_cand_embs.append(c_emb)
            padded_labels.append(l)
    
    # Stack and convert to tensors
    cand_embs = torch.tensor(np.stack(padded_cand_embs), device=device, dtype=torch.float32)
    labels = torch.tensor(np.array(padded_labels), device=device, dtype=torch.float32)
    
    return user_embs, cand_embs, labels

def evaluate_recommendations(model, dataloader, newsid_to_idx, news_emb_np, device, k=5):
    """
    Evaluate recommendation model using NDCG@k and MRR@k
    """
    model.eval()
    ndcgs = []
    mrrs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Evaluating NDCG@{k}"):
            user_embs, cand_embs, labels = collate_batch(batch, newsid_to_idx, news_emb_np, device)
            if user_embs.shape[0] == 0:
                continue
            
            logits = model(user_embs, cand_embs)
            scores = torch.sigmoid(logits).cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            for i in range(scores.shape[0]):
                # Calculate NDCG@k
                idx = np.argsort(-scores[i])[:k]
                if labels_np[i][idx].sum() > 0:
                    gains = labels_np[i][idx]
                    discounts = 1.0 / np.log2(np.arange(2, len(idx) + 2))
                    dcg = (gains * discounts).sum()
                    
                    # Calculate ideal DCG
                    ideal_idx = np.argsort(-labels_np[i])[:k]
                    ideal_gains = labels_np[i][ideal_idx]
                    idcg = (ideal_gains * discounts).sum()
                    
                    ndcg = dcg / idcg if idcg > 0 else 0.0
                    ndcgs.append(ndcg)
                else:
                    ndcgs.append(0.0)
                
                # Calculate MRR@k
                mrr = 0.0
                for j, idx in enumerate(np.argsort(-scores[i])[:k]):
                    if labels_np[i][idx] > 0:
                        mrr = 1.0 / (j + 1)
                        break
                mrrs.append(mrr)
    
    avg_ndcg = np.mean(ndcgs) if ndcgs else 0.0
    avg_mrr = np.mean(mrrs) if mrrs else 0.0
    
    return avg_ndcg, avg_mrr

def evaluate_recommendation_model(
    model_path, 
    news_path, 
    behaviors_path, 
    emb_dim=128,
    max_features=20000,
    max_history=50,
    k_values=[5, 10],
    output_path=None
):
    """
    Evaluate the news recommendation model on test data.
    
    Args:
        model_path: Path to the PyTorch model (.pth)
        news_path: Path to the news data in TSV format
        behaviors_path: Path to the behaviors data in TSV format
        emb_dim: Embedding dimension
        max_features: Maximum features for TF-IDF
        max_history: Maximum history length
        k_values: List of k values for NDCG@k and MRR@k
        output_path: Optional path to save results
        
    Returns:
        Dictionary of evaluation metrics
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING NEWS RECOMMENDATION MODEL")
    print(f"{'='*80}")
    
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if PyTorch model exists (prefer PyTorch over ONNX for evaluation)
    pth_path = model_path.replace('.onnx', '.pth')
    if not os.path.exists(pth_path):
        print(f"Error: PyTorch model file {pth_path} not found!")
        return None
    
    # Load data
    print(f"Loading news data from {news_path}")
    news_df = pd.read_csv(news_path, sep='\t', header=None, names=[
        'id', 'category', 'subcategory', 'title', 'abstract', 'url', 'title_entities', 'abstract_entities'
    ])
    
    # Extract news IDs and create a dictionary of news content
    news_ids = news_df['id'].tolist()
    news_content = news_df.apply(
        lambda row: f"{row['title']} {row['abstract']}", axis=1
    ).tolist()
    
    # Create news embeddings using TF-IDF + SVD
    print("Creating news embeddings...")
    news_emb_np = encode_text_svd(news_content, max_features=max_features, n_components=emb_dim)
    newsid_to_idx = {nid: i for i, nid in enumerate(news_ids)}
    
    # Load behaviors data
    print(f"Loading behaviors data from {behaviors_path}")
    dev_ds = MINDBehaviorDataset(
        behaviors_path, 
        set(news_ids), 
        max_history=max_history, 
        neg_sample_size=4, 
        sample_negatives=False
    )
    
    # Create DataLoader
    def custom_collate(batch):
        return batch
    
    dev_loader = torch.utils.data.DataLoader(
        dev_ds, 
        batch_size=128, 
        shuffle=False, 
        collate_fn=custom_collate
    )
    
    # Load model
    print(f"Loading model from {pth_path}")
    model = ScoringModel(emb_dim, hidden_dim=256).to(device)
    model.load_state_dict(torch.load(pth_path, map_location=device))
    model.eval()
    
    # Evaluate model
    print("Evaluating recommendations...")
    results = {}
    
    # Calculate metrics for each k value
    for k in k_values:
        ndcg, mrr = evaluate_recommendations(model, dev_loader, newsid_to_idx, news_emb_np, device, k=k)
        results[f'ndcg@{k}'] = ndcg
        results[f'mrr@{k}'] = mrr
        print(f"NDCG@{k}: {ndcg:.4f}, MRR@{k}: {mrr:.4f}")
    
    # Collect overall results
    evaluation_time = time.time() - start_time
    evaluation_results = {
        "model_type": "News Recommendation (Neural Network)",
        "model_path": pth_path,
        "news_data": news_path,
        "behaviors_data": behaviors_path,
        "metrics": results,
        "parameters": {
            "embedding_dimension": emb_dim,
            "max_features": max_features,
            "max_history": max_history,
            "k_values": k_values
        },
        "evaluation_time_seconds": evaluation_time
    }
    
    # Print overall results
    print(f"\nNews Recommendation Model Evaluation Results:")
    print(f"Number of news items: {len(news_ids)}")
    print(f"Number of behavior samples: {len(dev_ds)}")
    for k in k_values:
        print(f"NDCG@{k}: {results[f'ndcg@{k}']:.4f}")
        print(f"MRR@{k}: {results[f'mrr@{k}']:.4f}")
    print(f"Evaluation completed in {evaluation_time:.2f} seconds")
    
    # Save results if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        print(f"Results saved to {output_path}")
    
    return evaluation_results


def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Current timestamp for filenames
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Evaluate fake news detection model
    fake_news_results = None
    if not args.skip_fake_news:
        fake_news_output = os.path.join(args.output_dir, f"fake_news_evaluation_{timestamp}.json")
        fake_news_results = evaluate_fake_news_model(
            args.fake_news_model,
            args.fake_news_test_data,
            label_col=args.label_col,
            text_col=args.text_col,
            output_path=fake_news_output
        )
    
    # Evaluate news recommendation model
    recommendation_results = None
    if not args.skip_recommendation:
        recommendation_output = os.path.join(args.output_dir, f"recommendation_evaluation_{timestamp}.json")
        recommendation_results = evaluate_recommendation_model(
            args.recommendation_model,
            args.news_data,
            args.behaviors_data,
            emb_dim=args.emb_dim,
            max_features=args.max_features,
            max_history=args.max_history,
            k_values=args.k_values,
            output_path=recommendation_output
        )
    
    # Generate combined report
    if fake_news_results or recommendation_results:
        combined_output = os.path.join(args.output_dir, f"combined_evaluation_{timestamp}.json")
        combined_results = {
            "timestamp": timestamp,
            "fake_news_detection": fake_news_results,
            "news_recommendation": recommendation_results
        }
        
        with open(combined_output, 'w') as f:
            json.dump(combined_results, f, indent=2)
        
        print(f"\nCombined evaluation results saved to {combined_output}")
        
        # Generate text report for easy reading
        report_path = os.path.join(args.output_dir, f"evaluation_report_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write(f"DCIT316 Semester Project - Model Evaluation Report\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*80}\n\n")
            
            if fake_news_results:
                f.write(f"FAKE NEWS DETECTION MODEL EVALUATION\n")
                f.write(f"{'-'*50}\n")
                f.write(f"Model: {fake_news_results['model_path']}\n")
                f.write(f"Test data: {fake_news_results['test_data']}\n")
                f.write(f"Samples: {fake_news_results['samples_count']}\n")
                f.write(f"Classes: {', '.join(map(str, fake_news_results['classes']))}\n\n")
                
                f.write(f"Performance Metrics:\n")
                f.write(f"  Accuracy: {fake_news_results['metrics']['accuracy']:.4f}\n")
                f.write(f"  Precision (weighted): {fake_news_results['metrics']['precision']:.4f}\n")
                f.write(f"  Recall (weighted): {fake_news_results['metrics']['recall']:.4f}\n")
                f.write(f"  F1 Score (weighted): {fake_news_results['metrics']['f1_score']:.4f}\n")
                if fake_news_results['metrics']['auc']:
                    f.write(f"  AUC: {fake_news_results['metrics']['auc']:.4f}\n")
                f.write(f"\n")
                
                f.write(f"Class-wise Performance:\n")
                for cls in fake_news_results['classes']:
                    if str(cls) in fake_news_results['classification_report']:
                        cls_report = fake_news_results['classification_report'][str(cls)]
                        f.write(f"  Class '{cls}':\n")
                        f.write(f"    Precision: {cls_report['precision']:.4f}\n")
                        f.write(f"    Recall: {cls_report['recall']:.4f}\n")
                        f.write(f"    F1 Score: {cls_report['f1-score']:.4f}\n")
                        f.write(f"    Support: {cls_report['support']}\n")
                f.write(f"\n")
                
                f.write(f"Evaluation time: {fake_news_results['evaluation_time_seconds']:.2f} seconds\n\n")
            
            if recommendation_results:
                f.write(f"NEWS RECOMMENDATION MODEL EVALUATION\n")
                f.write(f"{'-'*50}\n")
                f.write(f"Model: {recommendation_results['model_path']}\n")
                f.write(f"News data: {recommendation_results['news_data']}\n")
                f.write(f"Behaviors data: {recommendation_results['behaviors_data']}\n\n")
                
                f.write(f"Performance Metrics:\n")
                for k in args.k_values:
                    f.write(f"  NDCG@{k}: {recommendation_results['metrics'][f'ndcg@{k}']:.4f}\n")
                    f.write(f"  MRR@{k}: {recommendation_results['metrics'][f'mrr@{k}']:.4f}\n")
                f.write(f"\n")
                
                f.write(f"Model Parameters:\n")
                f.write(f"  Embedding Dimension: {recommendation_results['parameters']['embedding_dimension']}\n")
                f.write(f"  Max Features: {recommendation_results['parameters']['max_features']}\n")
                f.write(f"  Max History: {recommendation_results['parameters']['max_history']}\n")
                f.write(f"\n")
                
                f.write(f"Evaluation time: {recommendation_results['evaluation_time_seconds']:.2f} seconds\n")
            
        print(f"Human-readable evaluation report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models for DCIT316 Semester Project")
    
    # Common arguments
    parser.add_argument("--output-dir", default="evaluation_results", help="Directory to save evaluation results")
    
    # Fake news model arguments
    parser.add_argument("--skip-fake-news", action="store_true", help="Skip fake news detection model evaluation")
    parser.add_argument("--fake-news-model", default="models/lr_pipeline.onnx", help="Path to fake news ONNX model")
    parser.add_argument("--fake-news-test-data", default="Data/liar_dataset/valid.tsv", help="Path to fake news test data")
    parser.add_argument("--label-col", type=int, default=1, help="Column index for label in fake news TSV")
    parser.add_argument("--text-col", type=int, default=2, help="Column index for text in fake news TSV")
    
    # Recommendation model arguments
    parser.add_argument("--skip-recommendation", action="store_true", help="Skip news recommendation model evaluation")
    parser.add_argument("--recommendation-model", default="models/recommender_scoring.onnx", help="Path to recommendation model")
    parser.add_argument("--news-data", default="Data/MINDsmall_dev/news.tsv", help="Path to news data")
    parser.add_argument("--behaviors-data", default="Data/MINDsmall_dev/behaviors.tsv", help="Path to behaviors data")
    parser.add_argument("--emb-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--max-features", type=int, default=20000, help="Maximum features for TF-IDF")
    parser.add_argument("--max-history", type=int, default=50, help="Maximum history length")
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10], help="k values for NDCG@k and MRR@k")
    
    args = parser.parse_args()
    main(args)