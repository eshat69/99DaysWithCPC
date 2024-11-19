import sys
R = lambda: map(int, next(sys.stdin).split())
t,=R()
while t:
    t-=1
    n,c= R()
    *s, = R()
    ans = c*(c+1)//2+c
    d=[0]*2
    for i in range(n):
        ans -= (s[i]//2+(c-s[i])+1)
        d[s[i]%2] +=1
    print(1 + ans + d[0]*(d[0]-1)//2 +  d[1]*(d[1]-1)//2)