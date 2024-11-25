# Read input
a, b, c, d = map(int, input().split())

# List of all restaurants
restaurants = {1, 2, 3, 4, 5}

# Find the restaurant not yet visited
visited = {a, b, c, d}
remaining = restaurants - visited

# Output the remaining restaurant number
print(remaining.pop())

