from sklearn.metrics import precision_score, recall_score, f1_score

# Example data (replace with your actual data)
y_true = [0, 1, 0, 0, 1, 1, 0, 1]
y_pred = [0, 1, 1, 0, 0, 1, 0, 1]

# Calculate precision and recall
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)

# Calculate F1-score
f1 = f1_score(y_true, y_pred)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-score: {f1:.4f}')
