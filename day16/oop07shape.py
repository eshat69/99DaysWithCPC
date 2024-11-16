class shape :
    def __init__(self,d1,d2):
        self.d1=d1
        self.d2=d2
    def area(self):
        print("area")
class tringle (shape):
    def area(self):
        area = 0.5*self.d1*self.d2
        print("area of tringle = ",area)
class rectangle (shape):
    def area(self):
        area = self.d1*self.d2
        print("area of rectangle = ",area)
class ellipse(shape) :
    def area(self):
        area = 3.1416 *self.d1*self.d2
        print("area of ellipse = ",area)
class circle(shape) :
    def area(self):
        area= 3.1416*self.d1*self.d2
        print("area of circle = ",area)
t1=tringle(10,20)
t1.area()
r1=rectangle(10,5)
r1.area()
el=ellipse(10,15)
el.area()
c1=circle(10,10 )#d1=d2 when circle
c1.area()