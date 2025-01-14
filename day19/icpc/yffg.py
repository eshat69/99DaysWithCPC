def fix_expressions(t, test_cases):
    results = []
    for s in test_cases:
        a = int(s[0])
        b = int(s[2])
        op = s[1]

        # Check if the current expression is valid
        if (op == '<' and a < b) or (op == '=' and a == b) or (op == '>' and a > b):
            results.append(s)  # Expression is already valid
        else:
            # Correct the expression
            # If digits are equal, use '='
            if a == b:
                results.append(f"{a}={b}")
            # If a < b, use '<'
            elif a < b:
                results.append(f"{a}<{b}")
            # If a > b, use '>'
            else:
                results.append(f"{a}>{b}")
    return results


# Input
t = int(input())
test_cases = [input().strip() for _ in range(t)]

# Solve and Output
output = fix_expressions(t, test_cases)
print("\n".join(output))
