"""
Load the model and test set, compute actual accuracy from scratch.
Verify the 95.4% claim is real.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd

print("="*80)
print("VERIFYING ACCURACY FROM SCRATCH")
print("="*80)

# 1. Load test data
print("\n1. Loading test data...")
test_df = pd.read_csv('data/frozen_splits/test.csv')
print(f"   Test samples: {len(test_df)}")
print(f"   Class distribution:")
print(f"   {test_df['label'].value_counts().to_string()}")

# 2. Load model and tokenizer
print("\n2. Loading model...")
model_path = 'models/final_backbone'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()
print(f"   Model loaded from: {model_path}")

# 3. Generate predictions from scratch
print("\n3. Generating predictions...")
# Combine product_name, brands, and ingredients_text for input text
# Map label names to integers
label_map = {'safe': 0, 'trace': 1, 'contain': 2}

texts = []
for _, row in test_df.iterrows():
    text = f"{row['product_name']} {row['brands']} {row['ingredients_text']}"
    texts.append(text)
labels = [label_map[label] for label in test_df['label']]

predictions = []
confidences = []

with torch.no_grad():
    for i, text in enumerate(texts):
        if (i + 1) % 50 == 0:
            print(f"   Progress: {i+1}/{len(texts)}")
        
        inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True)
        outputs = model(**inputs)
        logits = outputs.logits[0]
        
        # Apply temperature scaling
        T = 1.45  # Calibrated temperature
        probs = torch.softmax(logits / T, dim=0).numpy()
        
        pred = probs.argmax()
        conf = probs.max()
        
        predictions.append(pred)
        confidences.append(conf)

print(f"   Completed: {len(predictions)} predictions")

# 4. Compute metrics
print("\n4. Computing metrics...")
accuracy = accuracy_score(labels, predictions)
f1_macro = f1_score(labels, predictions, average='macro')

print("\n" + "="*80)
print("ACTUAL MODEL PERFORMANCE (from scratch)")
print("="*80)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Macro F1: {f1_macro:.4f}")
print(f"\nPaper claims: 95.4% accuracy")
print(f"Actual computed: {accuracy*100:.2f}%")
print(f"Match: {'YES' if abs(accuracy - 0.954) < 0.01 else 'NO'}")

print("\n" + "="*80)
print("DETAILED CLASSIFICATION REPORT")
print("="*80)
print(classification_report(labels, predictions, 
                           target_names=['safe', 'trace', 'contain']))

# 5. Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels, predictions)
print("\n" + "="*80)
print("CONFUSION MATRIX")
print("="*80)
print("         Predicted")
print("Actual   Safe  Trace  Contain")
print(f"Safe      {cm[0,0]:4d}  {cm[0,1]:5d}  {cm[0,2]:6d}")
print(f"Trace     {cm[1,0]:4d}  {cm[1,1]:5d}  {cm[1,2]:6d}")
print(f"Contain   {cm[2,0]:4d}  {cm[2,1]:5d}  {cm[2,2]:6d}")

# 6. Confidence statistics
print("\n" + "="*80)
print("CONFIDENCE STATISTICS")
print("="*80)
conf = np.array(confidences)
print(f"Mean confidence: {conf.mean():.4f}")
print(f"Median confidence: {np.median(conf):.4f}")
print(f"Min confidence: {conf.min():.4f}")
print(f"Max confidence: {conf.max():.4f}")
print(f"Std confidence: {conf.std():.4f}")

# 7. Save for later use
print("\n5. Saving results...")
results = pd.DataFrame({
    'text': texts,
    'true_label': labels,
    'pred_label': predictions,
    'confidence': confidences
})
results.to_csv('dist/full_test_predictions.csv', index=False)
print("   Saved to: dist/full_test_predictions.csv")

# 8. Compare with saved predictions
print("\n6. Comparing with saved predictions...")
saved_preds = pd.read_csv('results/verified_final/test_preds.csv')
saved_labels = saved_preds['label'].values
saved_pred = saved_preds['prediction'].values

agreement = (np.array(predictions) == saved_pred).mean()
print(f"   Agreement with saved predictions: {agreement:.4f} ({agreement*100:.2f}%)")

if agreement == 1.0:
    print("   [OK] Perfect agreement - predictions match saved file")
else:
    print("   [WARNING] Mismatch between fresh and saved predictions!")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)

