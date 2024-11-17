def count_multiples(x, l, r):
    # Counts how many multiples of x are within the range [l, r]
    return (r // x) - ((l - 1) // x)

def max_operations(l, r, k):
    # Initialize count of operations
    operations = 0
    for x in range(l, r + 1):
        # Check if x has at least k multiples in [l, r]
        if count_multiples(x, l, r) >= k:
            operations += 1
    return operations

# Read input and process multiple test cases
t = int(input())
results = []
for _ in range(t):
    l, r, k = map(int, input().split())
    results.append(max_operations(l, r, k))

# Output the results for each test case
print("\n".join(map(str, results)))
