import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return 0.001*x**4-0.2*x**3-0.3*x**2+x

def calculate(low,up):
    if f(low)<f(up):
        while True:
            mid = (low+up)/2
            if f(up)>0:
                up = mid
            if f(low)<0:
                low = mid
            if abs(low-up) < 0.000000000001:
                break
    else:
        while True:
            mid = (low+up)/2
            if f(up)<0:
                up = mid
            if f(low)>0:
                low = mid
            if abs(low-up) < 0.000000000001:
                break
    return low,up


# 记录交点区间的位置
xu = []
xl = []
# 区间
x_low = -5
x_up = 5
#步长
delx = 0.001
# 初始值
x_i = x_low

while x_i<=x_up:
    if (f(x_i)>0 and f(x_i+delx)<0) or (f(x_i)<0 and f(x_i+delx)>0):
        xu.append(x_i+delx)
        xl.append(x_i)
        # print("xl = ",x_i)
        # print("xu = ",x_i+delx)
        # print("-"*60)
    x_i+=delx

# 通过二分法细算
print("分点后")
print("xl = " , xl)
print("xu = " , xu)
print("-"*60)
# 结果
resultl = []
resultu = []

for i in range(len(xl)):
    resultl.append(f(xl[i]))
    resultu.append(f(xu[i]))

print("分点后的结果")
print("result_l = ",resultl)
print("result_u = ",resultu)
print("-"*60)

u = 0
l = 0

x_l=[]
x_u=[]

if len(xl) !=0:
    for i in range(len(xl)):
        l,u = calculate(xl[i],xu[i])
        x_l.append(l)
        x_u.append(u)

# 细算后
print("细算后")
print("x_l = ",x_l)
print("x_u = ",x_u)
print("-"*60)

result_l = []
result_u = []
for i in range(len(x_l)):
    result_l.append(f(x_l[i]))
    result_u.append(f(x_u[i]))

print("二分法后的结果")
print("result_l = ",result_l)
print("result_u = ",result_u)
print("-"*60)



# 画图
# 生成有序序列
x = np.linspace(-5,5,1000000)

plt.plot(x,f(x),"r-")
plt.plot(xl,f(np.array(xl)),"yH")
plt.plot(xu,f(np.array(xu)),"yv")
plt.plot(x_l,f(np.array(x_l)),"gH")
plt.plot(x_u,f(np.array(x_u)),"r*")
plt.plot(x,x*0,"g-.")
plt.legend(["function_line",'before_lower_x',"before_upper_x","after_lower_x","after_upper_x","line_is_0"])
plt.show()
