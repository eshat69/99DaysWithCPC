# Example data: True positives and false negatives
true_positives = 120
false_negatives = 40

# Calculate recall
recall = true_positives / (true_positives + false_negatives)

# Print the recall
print(f'Recall: {recall:.2f}')
