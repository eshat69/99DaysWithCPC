from collections import Counter
def solve_diy(test_cases):
    results = []
    for n, arr in test_cases:
        count = Counter(arr)
        candidates = []
        for num, freq in count.items():
            if freq >= 2:
                candidates.append(num)
        candidates.sort()

        if len(candidates) < 2:
            results.append("NO")
        else:
            x1 = candidates[0]
            x2 = candidates[-1]
            results.append("YES")
            results.append(f"{x1} {x1} {x1} {x2} {x2} {x1} {x2} {x2}")

    return results
t = int(input())
test_cases = []
for _ in range(t):
    n = int(input())
    arr = list(map(int, input().split()))
    test_cases.append((n, arr))
results = solve_diy(test_cases)
print("\n".join(results))
