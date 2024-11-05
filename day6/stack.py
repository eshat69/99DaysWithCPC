class stack :
    def __init__(self):
        self.value=[];
    def push(self,s):
        self.value = [s]+self.value
    def pop(self):
        return self.value.pop(0)
s=stack()
s.push(10)
s.push(20)
s.push(30)
print(s.value)
s.pop()
print(s.value)
s2 = stack()
for i in range(1, 5):
    s2.push(i)
print(s2.value)
s2.pop()
print(s2.value)