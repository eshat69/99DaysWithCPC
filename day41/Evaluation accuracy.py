def calculate_accuracy(true_positives, true_negatives, total_predictions):
    # Calculate accuracy
    accuracy = (true_positives + true_negatives) / total_predictions
    return accuracy

# Example usage
true_positives = 85
true_negatives = 60
total_predictions = 100

accuracy = calculate_accuracy(true_positives, true_negatives, total_predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")
