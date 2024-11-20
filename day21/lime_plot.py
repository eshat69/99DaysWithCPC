import matplotlib.pyplot as plt
x = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
y1 = [200, 104, 278, 43, 200, 150, 199]
y2 = [1000, 990, 800, 900, 1009, 1000, 600]

plt.plot(x, y1,label="profit")
plt.plot(x, y2, marker="*", color="blue", label="invest",ls="--", alpha=0.5)
plt.legend()
plt.show()
