NumOfWord=0
NumOfLetter=0
NumOfDigit=0
list=input("enter a text :")
for x in list :
    x=x.lower()
    if x>='a' and x<='z' :
        NumOfLetter=NumOfLetter+1
    elif x >='0' and x<='9' :
        NumOfDigit=NumOfDigit+1
    elif x==' ' :
        NumOfWord=NumOfWord+1
print(list, "has ")
print(NumOfLetter,"letters",NumOfDigit,"digits",NumOfWord+1,"words")