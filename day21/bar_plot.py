import matplotlib.pyplot as plt
x = ["Fifty Shades of Grey", "Fifty Shades Darker", "Fifty Shades Freed","365 days","my fault","after","beautiful disaster","no hard feelings","knock knock",]
y = [341000, 97000, 254000,320000,256000,188000,110000,324550,120000]
color = ["red", "purple", "pink", "blue", "black", "navy", "violet", "yellow", "orange"]
plt.bar(x, y, color=color[:len(x)])
plt.xlabel("Movies")
plt.ylabel("Rated",fontsize= 6)
plt.title("18+ movies",fontsize=20)
plt.xticks(rotation=20, ha="right")  # Rotate x-axis labels for better readability
plt.show()
