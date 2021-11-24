import numpy as np
from math import *
import math
############################函数板块
customers = [5000,2000,500]
total_invest = 1000000
cost_11 = 200
cost_87 = 700

n_87  = 0

# 计算比率
def cal_sold_rate_11(price_rate_11):
    if price_rate_11 <0.9:
        rate = 1.0;
    if price_rate_11 >=0.9 and price_rate_11<=1.3:
        rate = cos((price_rate_11-0.9)/0.6/2*pi)
    if price_rate_11 >1.3:
        rate = 0.0
    return rate
# 计算比率
def cal_sold_rate_87(price_rate_87):
    if price_rate_87<0.9:
        rate = 0.3
    if price_rate_87>=0.9 and price_rate_87<=1.5:
        rate = 0.3*cos((price_rate_87-0.9)/0.6*pi/2)
    if price_rate_87>1.5:
        rate = 0
    return rate

# 计算盈利
def cal_profit(num_11,price_rate_11,price_rate_87):
    num_87 = math.floor((total_invest-num_11*cost_11)/(cost_87))
    global n_87 
    n_87= num_87

    sold_rate_11 = cal_sold_rate_11(price_rate_11)
    sold_price_11 = price_rate_11*cost_11
    sold_num_11 = customers[0] * sold_rate_11


    sold_rate_87 = cal_sold_rate_87(price_rate_87)
    sold_price_87 = price_rate_87*cost_87
    sold_num_87 = customers[1] * sold_rate_87

    
    if price_rate_11 <= price_rate_87:
        sold_num_11+=customers[2]
    else:
        sold_num_87+=customers[2]
    
    if sold_num_11 > num_11:
        sold_num_11 = num_11
    if sold_num_87 > num_87:
        sold_num_87 = num_87
        
    profit = sold_num_11 * (sold_price_11-cost_11)  +  sold_num_87 * (sold_price_87 - cost_87)
    return profit,sold_num_11,sold_num_87,price_rate_11,price_rate_87

    

# 调用盈利
def F(num,prince_rate_11,price_rate_87):
    result = []
    for i in range(len(num)):
        result.append(cal_profit(num[i],prince_rate_11[i],price_rate_87[i])[0])
    return result


############################函数板块

############################遗传算法板块
DNA_SIZE = 24
POP_SIZE = 1000
CORSSOVER_RATE = 0.8
MUTATION_RATE = 0.005
N_GENERATIONS = 50
X_BOUND = [1000,5000]
Y_BOUND = [1.0,1.31]
Z_BOUND = [1.0,1.51]

# 获取最优解
def get_fitness(pop):
    x,y,z = translateDNA(pop)
    pred = F(x,y,z)
    return np.array(pred)

# DNA转换
def translateDNA(pop):
    x_pop = pop[:, ::3]
    y_pop = pop[:,1::3]
    z_pop = pop[:,2::3]

    x = x_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(X_BOUND[1]-X_BOUND[0])+X_BOUND[0]
    y = y_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(Y_BOUND[1]-Y_BOUND[0])+Y_BOUND[0]
    z = z_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(Z_BOUND[1]-Z_BOUND[0])+Z_BOUND[0]
    x = list(map(int,x))
    return x,y,z

# 交叉和变异
def corssover_and_mutation(pop,CORSSOVER_RATE = 0.8):
    new_pop = []
    for father in pop:
        child = father
        if np.random.rand()<CORSSOVER_RATE:
            mother = pop[np.random.randint(POP_SIZE)]
            cross_point = np.random.randint(low = 0,high = DNA_SIZE*3)
            child[cross_point:] = mother[cross_point:]
        mutation(child)
        new_pop.append(child)
    return new_pop

# 变异
def mutation(child,MUTATION_RATE = 0.005):
    if np.random.rand() < MUTATION_RATE:
        mutate_point = np.random.randint(0,DNA_SIZE*3)
        child[mutate_point] = child[mutate_point]^1
# 随机选择
def select(pop, fitness):
    idx = np.random.choice(np.arange(POP_SIZE),size = POP_SIZE,replace = True,p = (fitness)/(fitness.sum()))
    
    return pop[idx]

# 打印信息
def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:",fitness[max_fitness_index])
    x,y,z = translateDNA(pop);
    print("the best geen is :",pop[max_fitness_index])
    print("11的数量: ",x[max_fitness_index],"\t11的收益: ",y[max_fitness_index],"\n87的数量: ",n_87,"\t87的收益: ",z[max_fitness_index])


# 随机生成
pop  = np.random.randint(2,size=(POP_SIZE,DNA_SIZE*3))
# 更新迭代
for _ in range(N_GENERATIONS):
    x,y,z = translateDNA(pop)
    pop = np.array(corssover_and_mutation(pop,CORSSOVER_RATE))
    fitness = get_fitness(pop)
    pop= select(pop,fitness)
print_info(pop)
###########################遗传算法板块



########################### 网格法
profit_opt = 0
num_11_best = 0
num_87_best = 0
price_rate_11_opt = 1.0
price_rate_87_opt = 1.0

for mun_11 in range(1000,5000):
    for ind_11 in range(61):
        for ind_87 in range(51):
            profit,sold_mun_11,sold_num_87,price_rate_11,price_rate_87 = cal_profit(mun_11,1+0.01*ind_11,1.0+0.01*ind_87)
            if profit>profit_opt:
                profit_opt = profit
                num_11_best = sold_mun_11
                num_87_best = sold_num_87
                price_rate_11_opt = price_rate_11
                price_rate_87_opt = price_rate_87 
print(profit_opt,num_11_best,num_87_best,price_rate_11_opt,price_rate_87_opt)