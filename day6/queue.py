class queue :
    def __init__(self):
        self.value=[]
    def enqueue(self,q):
        self.value.append(q)
    def dequeue(self):
        front=self.value[0]
        self.value=self.value[1:]
        return front
q1=queue()
q1.enqueue(1)
q1.enqueue(2)
print(q1.value)
q1.dequeue()
print(q1.value)
q2 = queue()
for i in range(5):
    q2.enqueue(i)
print(q2.value)
q2.dequeue()
print(q2.value)
q2.dequeue()
print(q2.value)