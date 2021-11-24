import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

DNA_SIZE = 24
POP_SIZE = 50
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.005
N_GENERATIONS = 200
X_BOUND = [-3,3]
Y_BOUND = [-3,3]


# 定义适应度函数
def F(x,y):
    return 3 * (1-x) ** 2 * np.exp( -(x**2) - (y+1)**2) -\
        10 * ( x/5 - x**3 - y**5 ) *np.exp( -x**2 - y**2 )-\
            1/3**np.exp( -(x + 1)**2 - y**2 )

# 绘图
def plot_3d(ax):
    X = np.linspace(*X_BOUND,100)
    Y = np.linspace(*Y_BOUND,100)
    X,Y = np.meshgrid(X,Y)
    Z = F(X,Y)
    ax.plot_surface(X,Y,Z,rstride = 1,cstride = 1,cmap = cm.coolwarm)
    ax.set_zlim(-10,10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.pause(1)
    plt.show()

# 定义计算适应度值得函数和转码函数
def get_fitness(pop):
    x,y = translateDNA(pop)
    # print(x,y)
    pred = F(x,y)
    return (pred - np.min(pred)) + 1e-3

def translateDNA(pop):
    x_pop = pop[:,1::2]
    y_pop = pop[:,::2]

   # print(x_pop)
    #print(y_pop)
    x = x_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(X_BOUND[1]-X_BOUND[0])+X_BOUND[0]
    y = y_pop.dot(2**np.arange(DNA_SIZE)[::-1])/float(2**DNA_SIZE-1)*(Y_BOUND[1]-Y_BOUND[0])+Y_BOUND[0]
    return x,y

# 定义选择、交叉和变异操作函数
def crossover_and_mutation(pop,CROSSOVER_RATE = 0.8):
    new_pop = []
    for father in pop:
        child = father
        if np.random.rand()<CROSSOVER_RATE:
            mother = pop[np.random.randint(POP_SIZE)]
            cross_points = np.random.randint(low = 0,high = DNA_SIZE*2)
            child[cross_points:] = mother[cross_points:]
        mutation(child)
        new_pop.append(child)
    return new_pop

def mutation(child,MUTATION_RATE = 0.003):
    if np.random.rand()<MUTATION_RATE:
        mutate_point = np.random.randint(0,DNA_SIZE*2)
        child[mutate_point] = child[mutate_point]^1

def select(pop,fitness):
    idx = np.random.choice(np.arange(POP_SIZE),size = POP_SIZE,replace = True,p = (fitness)/(fitness.sum()))
    return pop[idx]

def print_info(pop):
    fitness = get_fitness(pop)
    max_fitness_index = np.argmax(fitness)
    print("max_fitness:",fitness[max_fitness_index])
    x,y = translateDNA(pop)
    print("最优的基因型:",pop[max_fitness_index])
    print("(x , y):",(x[max_fitness_index],y[max_fitness_index]))


fig = plt.figure()
ax = Axes3D(fig)
plt.ion()
plot_3d(ax)

pop = np.random.randint(2,size = (POP_SIZE,DNA_SIZE*2))

for _ in range(N_GENERATIONS):
    x,y = translateDNA(pop)
    if "sca" in locals():
        sca.remove()
    sca = ax.scatter(x,y,F(x,y),c = "black",marker = "o")
    plt.show()
    plt.pause(0.1)
    pop = np.array(crossover_and_mutation(pop,CROSSOVER_RATE))
    fitness = get_fitness(pop)
    pop = select(pop,fitness)

print_info(pop)
plt.ioff()
plot_3d(ax)
