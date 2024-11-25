
for _ in range(int(input())):
    print(sum(3*(x // 2) + (x % 2)*(i < 3) for (i, x) in enumerate(map(int, input().split()))) // 3)
for _ in range(int(input())):
    # Read input: p1, p2, p3, p4 represent counts of 1s, 2s, 3s, and 4s
    p1, p2, p3, p4 = map(int, input().split())

    # Total number of games
    total_numbers = p1 + p2 + p3 + p4

    # XOR contribution of all numbers
    xor_sum = (1 * p1) ^ (2 * p2) ^ (3 * p3) ^ (4 * p4)

    # Bob wins when XOR is zero
    bob_wins = 0
    current_numbers = [1] * p1 + [2] * p2 + [3] * p3 + [4] * p4

    for i in range(total_numbers):
        # Try removing each number optimally
        for num in current_numbers:
            current_xor = xor_sum ^ num  # XOR without this number
            if current_xor == 0:
                bob_wins += 1
