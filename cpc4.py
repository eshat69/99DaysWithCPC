class tringle :
    def __init__(self,hight,base):
        self.hight = hight
        self.base = base
    def cal_area(self):
        area=0.5*self.hight*self.base
        print("area of tringle is ",area)
t1 = tringle(10,20)
t1.cal_area()
t2 = tringle(20,30)
t2.cal_area()
class squre :
    def __init__(self,base):
        self.base=base
    def cal_area(self):
        area= self.base*self.base
        print("area of squre is ",area)
s1 = squre(10)
s1.cal_area()
class circle :
    def __init__(self,redius):
        self.redius=redius
    def cal_area(self):
        area = 3.1416*self.redius*self.redius
        print("area of circle is  ",area)
c1 = circle(10)
c1.cal_area()