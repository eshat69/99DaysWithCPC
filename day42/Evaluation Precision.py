def calculate_precision(true_positives, false_positives):
    if true_positives + false_positives == 0:
        return 0  # to avoid division by zero
    precision = true_positives / (true_positives + false_positives)
    return precision

# Example usage
true_positives = 80
false_positives = 20
precision = calculate_precision(true_positives, false_positives)
print(f'Precision: {precision:.2f}')
