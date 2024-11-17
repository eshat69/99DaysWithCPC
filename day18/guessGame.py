from random import randint
for i in range (1,6):
    guess=int(input("Guess a number between 1 and 5: "))
    random=randint(1,5)
    if (guess== random) :
        print("you guessed correctly ")
    else :
        print("you guess incurrectly ")
        print(random)