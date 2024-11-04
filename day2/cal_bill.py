km = int(input("kilometer covered "))
if km <= 10 :
    bill = km*11
elif 10 < km <= 100 :
    bill = 10*11 + (km - 10)*10
else :
    bill = 10*11 + 90*10 + (km - 100)*9
print("Bill is",bill)