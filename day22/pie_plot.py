import matplotlib.pyplot as plt
brand = ["Apple", "OPPO", "Samsung", "Vivo", "Xiaomi", "OnePlus", "Realme", "Huawei"]
x = [18.12, 11, 25, 10, 8, 10, 4, 7]
c = ["green", "red", "yellow", "blue", "pink", "magenta", "orange", "purple"]
ex = [0.1, 0, 0, 0, 0, 0, 0, 0]
plt.pie(x, labels=brand, colors=c, explode=ex,shadow=True,autopct="%.2f",  startangle=140)
plt.title("Brand Market Share")
plt.show()
