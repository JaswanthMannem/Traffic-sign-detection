import random
x=random.randint(1,100)
print(x)
while(1):
    print("enter number")
    y=int(input())
    if y==x:
        print("correct")
        break
    elif y<x:
        print("number is higher")
    elif y>x:
        print("number is lower")