import numpy as np
a=np.array([12,4,5,87,99])
ar=np.array([2,3,4])
print(np.sum(a))
print(np.min(a))
print(np.max(a))
print(np.size(a))
print(np.mean(a))
print("cum sum ",np.cumsum(a)) #12   12+4=16   16+5=21     21+87=108      108+99=207
print(np.cumprod(ar))  #2       2*3=6         6*4=24

b = np.array([2, 1, 1, 5, 1]) 
price = np.size(a)  # Scalar value for the size of `a`
price_and_quantity = np.insert(b, 0, price)
print("Cumulative product of price and quantity:", np.cumprod(price_and_quantity))