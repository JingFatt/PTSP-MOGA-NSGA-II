import vrp8
import numpy as np
import random
from deap import base, creator, tools, algorithms
from vrp8 import cities
import sys
import matplotlib.pyplot as plt
import os
import io

class Tee(io.StringIO):
    """同时打印到Terminal和StringIO"""
    def __init__(self):
        super().__init__()
        self.stdout = sys.__stdout__  # 保存真正的stdout

    def write(self, s):
        self.stdout.write(s)   # 写到Terminal
        super().write(s)       # 写到内存

# Directory for saving results
def_save = 'results'
os.makedirs(def_save, exist_ok=True)

# --- 重定向 ---
buffer = Tee()
sys.stdout = buffer

# print(sys.executable)  # 打印当前 Python 解释器路径

# --- 导入数据 ---
# Extract coordinates
coords = np.array([[c['x'], c['y']] for c in vrp8.cities])
distance_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
print("Distance matrix shape:", distance_matrix.shape)
print("Top-left 5x5 block:\n", np.round(distance_matrix[:5, :5], 2))

# 假设 vrp8.py 已经在同目录，包含 cities 列表

# --- 参数设置 ---

NUM_SALESMEN = 5      # 销售员数量，可按需调整
POP_SIZE = 150        # 种群规模
NGEN = 100             # 迭代世代数
CXPB = 0.7            # 交叉概率
MUTPB = 0.2           # 变异概率

# --- 构建节点列表（加入一个 depot）+ 计算距离矩阵 ---
depot = {"id": 0, "x": 0, "y": 0, "demand": 0}
nodes = [depot] + cities
N = len(nodes)
coords = np.array([[n['x'], n['y']] for n in nodes])
distance_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

# --- DEAP 多目标设置 ---
creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMulti)
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, lambda: random.sample(range(0, N-1), N-1))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# --- 解码与评估函数 ---
def evaluate(ind):
    # 将访问顺序分配给各销售员（等分切分）
    k, m = NUM_SALESMEN, len(ind)
    chunk_size = m // k
    mapped = [g + 1 for g in ind]
    routes = []
    for i in range(NUM_SALESMEN):
            start = i*chunk_size
            end = (i+1)*chunk_size if i < NUM_SALESMEN-1 else len(mapped)
            seq = mapped[start:end]
            routes.append([0] + seq + [0])  # 0 是 depot
    
    # 计算每条路线的总距离与总需求
    distances, demands = [], []
    for r in routes:
        # 路程
        d = sum(distance_matrix[r[i]][r[i+1]] for i in range(len(r)-1))
        distances.append(d)
        # 需求
        dem = sum(nodes[idx]['demand'] for idx in r)
        demands.append(dem)
    
    total_dist = sum(distances)
    demand_diff = max(demands) - min(demands)
    return total_dist, demand_diff

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxPartialyMatched)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
toolbox.register("select", tools.selNSGA2)    

def nsga2(population, ngen, stats, hof):
     pop = population
     pop, log = algorithms.eaMuPlusLambda(
          pop, toolbox,
          mu=len(pop), lambda_=2*len(pop),
          cxpb=CXPB, mutpb=MUTPB,
          ngen=ngen, 
          stats=stats, 
          halloffame=hof,
          verbose=True
          )
     return pop, log

def drawMap(city_list, routes, title="Salesmen Routes"):
    """
    city_list: list of (id, x, y)，例如 [(0, 0, 0), (1, 2, 3), ...]
    routes: list of list，每个子列表是一条路径，例如 [[0, 2, 5, 0], [0, 3, 1, 0], ...]
    """
    plt.figure(figsize=(8, 6))
    # 绘制城市点
    for city in city_list:
        plt.plot(city[1], city[2], "ro")
        plt.annotate(str(city[0]), (city[1], city[2]))

    # 绘制路径，每条路径一个颜色
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta']
    for idx, route in enumerate(routes):
        route_coords = [(city_list[i][1], city_list[i][2]) for i in route]
        xs, ys = zip(*route_coords)
        plt.plot(xs, ys, marker='o', color=colors[idx % len(colors)], label=f"Salesman {idx+1}")

    plt.title(title)
    plt.legend()
    plt.grid(True)
    save_title = title.replace(' ', '_') + '.png'  # 把空格转成_
    save_path = os.path.join(def_save, save_title)
    plt.savefig(save_path)
    plt.show()

# --- Metrics 计算函数 ---
def hypervolume_2d(front, ref_point):
    """
    计算二维 Hypervolume（最小化问题）。
    front: numpy array of shape (n,2)
    ref_point: tuple (r1, r2)
    """
    if len(front) == 0:
        return 0.0
    pts = front[np.argsort(front[:,0])]
    hv = 0.0
    prev_f1 = ref_point[0]
    for f1, f2 in pts:
        width = max(prev_f1 - f1, 0)
        height = max(ref_point[1] - f2, 0)
        hv += width * height
        prev_f1 = f1
    return hv

def spacing(front):
    """
    Spacing：计算每个解到其最近邻的距离，然后返回这些最小距离的标准差
    """
    n = len(front)
    if n < 2:
        return 0.0
    ds = []
    for i in range(n):
        others = np.delete(front, i, axis=0)
        dists = np.linalg.norm(others - front[i], axis=1)
        ds.append(np.min(dists))
    return np.std(ds, ddof=1)

def range_spread(front):
    """
    简易 Spread：目标1和目标2的范围之和
    """
    if len(front) == 0:
        return 0.0
    return (front[:,0].max() - front[:,0].min()) + (front[:,1].max() - front[:,1].min())

def pop_diversity(population):
    """
    你的种群多样性定义：所有个体之间适应度距离的均值
    """
    if not population:
        return 0.0
    vals = np.array([ind.fitness.values for ind in population])
    dists = []
    for i in range(len(vals)):
        for j in range(i+1, len(vals)):
            dists.append(np.linalg.norm(vals[i] - vals[j]))
    return np.mean(dists) if dists else 0.0


# --- 主程序 ---
def main():
    pop = toolbox.population(n=POP_SIZE)

    # 先进行一次非支配排序，重要于 NSGA-II
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    pop = toolbox.select(pop, len(pop))
    initial_fitness = np.array([ind.fitness.values for ind in pop])
    
    # 记录 hall of fame（Pareto 前沿）
    hof = tools.ParetoFront()
    
    # 统计信息
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min, axis=0)
    stats.register("avg", np.mean, axis=0)
    stats.register("max", np.max, axis=0)
    
    # 存储
    logbook = tools.Logbook()
    logbook.header = ["gen","nevals","min","avg","max"]
    populations = [pop]

    # 迭代
    for gen in range(1, NGEN+1):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=CXPB, mutpb=MUTPB)
        for ind in offspring: ind.fitness.values = toolbox.evaluate(ind)
        pop = toolbox.select(pop + offspring, POP_SIZE)
        hof.update(pop)
        rec = stats.compile(pop)
        rec.update({"gen":gen, "nevals":len(offspring)})
        logbook.record(**rec)
        populations.append(pop)

    # --- 1. 画最初始地标分布（城市坐标，不连线） ---
    city_list = [(n['id'], n['x'], n['y']) for n in nodes]
    plt.figure(figsize=(8, 6))
    for city in city_list:
        plt.plot(city[1], city[2], "ro")
        plt.annotate(str(city[0]), (city[1], city[2]))
    plt.title("City Locations")
    plt.grid(True)
    title = plt.gca().get_title()
    save_title = title.replace(' ', '_') + '.png'
    save_path = os.path.join(def_save, save_title)
    plt.savefig(save_path)
    plt.show()

    # --- 2. 画初始种群中第一个解的路径 ---
    def decode(ind):
        k, m = NUM_SALESMEN, len(ind)
        chunk_size = m // k
        mapped = [g + 1 for g in ind]
        routes = []
        for i in range(NUM_SALESMEN):
            start = i * chunk_size
            end = (i+1) * chunk_size if i < NUM_SALESMEN - 1 else len(mapped)
            seq = mapped[start:end]
            routes.append([0] + seq + [0])
        return routes

    first_solution = pop[0]
    initial_routes = decode(first_solution)
    drawMap(city_list, initial_routes, title="Salesman Routes Before Optimization")

    # 运行 NSGA-II
    pop, log = nsga2(pop, NGEN, stats, hof)
    
    # --- 绘制一个Pareto前沿的解的路径 ---
    best_solution = hof[0]  # 选 Pareto 集合里第一个解来画
    routes = decode(best_solution)

    # 准备城市列表：[(id, x, y), ...]
    city_list = [(n['id'], n['x'], n['y']) for n in nodes]
    drawMap(city_list, routes, title = "Salesman Routes After Optimization") # 调用 drawMap

    # 从 logbook 中提取数据
    gen     = np.array(log.select("gen"))
    min_vals= np.array(log.select("min"))
    avg_vals= np.array(log.select("avg"))
    max_vals= np.array(log.select("max"))

    # --- 准备参考点 (ref_point) 用于 HV ---
    # 我们取初末代的最差值 10% 余量
    ref = (max(max_vals[:,0].max(), initial_fitness[:,0].max())*1.1,
           max(max_vals[:,1].max(), initial_fitness[:,1].max())*1.1)
    
    # 计算 metrics
    HV, SP, RS, DV, PS = [],[],[],[],[]
    for pop_i in populations:
        front = tools.sortNondominated(pop_i, k=len(pop_i), first_front_only=True)[0]
        fv = np.array([ind.fitness.values for ind in front])
        HV.append(hypervolume_2d(fv,ref))
        SP.append(spacing(fv))
        RS.append(range_spread(fv))
        DV.append(pop_diversity(pop_i))
        PS.append(len(front))

# ==== Graphs====================

    # 绘图：目标1（总距离）随代数演化 ---
    plt.figure()
    plt.plot(gen, min_vals[:,0], label="min")
    plt.plot(gen, avg_vals[:,0], label="avg")
    plt.plot(gen, max_vals[:,0], label="max")
    plt.xlabel("Generation")
    plt.ylabel("Total Distance")
    plt.title("Evolution of Total Distance")
    plt.legend()
    plt.grid(True)
    title = plt.gca().get_title()
    save_title = title.replace(' ', '_') + '.png'
    save_path = os.path.join(def_save, save_title)
    plt.savefig(save_path)
    plt.show()

    # 绘图：目标2（需求差）随代数演化 ---
    plt.figure()
    plt.plot(gen, min_vals[:,1], label="min")
    plt.plot(gen, avg_vals[:,1], label="avg")
    plt.plot(gen, max_vals[:,1], label="max")
    plt.xlabel("Generation")
    plt.ylabel("Demand Difference")
    plt.title("Evolution of Demand Difference")
    plt.legend()
    plt.grid(True)
    title = plt.gca().get_title()
    save_title = title.replace(' ', '_') + '.png'
    save_path = os.path.join(def_save, save_title)
    plt.savefig(save_path)
    plt.show()

    # 绘图：初始种群 vs 最终 Pareto 前沿 ---
    plt.figure()
    plt.scatter(initial_fitness[:,0], initial_fitness[:,1],
                c='gray', alpha=0.5, label="Initial Pop")
    final_vals = np.array([ind.fitness.values for ind in hof])
    plt.scatter(final_vals[:,0], final_vals[:,1],
                c='red', label="Pareto Front")
    plt.xlabel("Total Distance")
    plt.ylabel("Demand Difference")
    plt.title("Initial Population vs Final Pareto Front")
    plt.legend()
    plt.grid(True)
    title = plt.gca().get_title()
    save_title = title.replace(' ', '_') + '.png'
    save_path = os.path.join(def_save, save_title)
    plt.savefig(save_path)
    plt.show()

    # 绘制每代Pareto前沿的解
    plt.figure()
    for gen_num in range(len(log.select("gen"))):
        front = np.array([ind.fitness.values for ind in hof])
        plt.scatter(front[:, 0], front[:, 1], label=f"Generation {gen_num}")
    plt.xlabel("Total Distance")
    plt.ylabel("Demand Difference")
    plt.title("Pareto Front Evolution")
    # plt.legend()
    plt.grid(True)
    title = plt.gca().get_title()
    save_title = title.replace(' ', '_') + '.png'
    save_path = os.path.join(def_save, save_title)
    plt.savefig(save_path)
    plt.show()

    # 绘图：最优解随代数变化
    plt.figure()
    plt.plot(gen, min_vals[:, 0], label="Min Total Distance")
    plt.plot(gen, min_vals[:, 1], label="Min Demand Difference")
    plt.xlabel("Generation")
    plt.ylabel("Objective Value")
    plt.title("Convergence of Objectives")
    plt.legend()
    plt.grid(True)
    title = plt.gca().get_title()
    save_title = title.replace(' ', '_') + '.png'
    save_path = os.path.join(def_save, save_title)
    plt.savefig(save_path)
    plt.show()
    
    # --- 绘制 metrics 曲线 ---
    plt.figure(figsize=(10,5))
    plt.plot(gen, HV, '-o', markevery=[0, -1], label="Hypervolume")
    plt.xlabel("Generation"); plt.ylabel("HV"); plt.title("Hypervolume Over Generations")
    plt.grid(True); plt.legend(); 
    title = plt.gca().get_title()
    save_title = title.replace(' ', '_') + '.png'
    save_path = os.path.join(def_save, save_title)
    plt.savefig(save_path)
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(gen, SP,    label="Spacing")
    plt.plot(gen, RS, label="Range Spread")
    plt.xlabel("Generation"); plt.ylabel("Value"); plt.title("Spacing & Spread Over Generations")
    plt.grid(True); plt.legend(); 
    title = plt.gca().get_title()
    save_title = title.replace(' ', '_') + '.png'
    save_path = os.path.join(def_save, save_title)
    plt.savefig(save_path)
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(gen, DV, label="Population Diversity")
    plt.xlabel("Generation"); plt.ylabel("Diversity"); plt.title("Diversity Over Generations")
    plt.grid(True); plt.legend(); 
    title = plt.gca().get_title()
    save_title = title.replace(' ', '_') + '.png'
    save_path = os.path.join(def_save, save_title)
    plt.savefig(save_path)
    plt.show()

    plt.figure(figsize=(10,5))
    plt.plot(gen, PS, label="Pareto Front Size")
    plt.xlabel("Generation"); plt.ylabel("Number of Solutions"); plt.title("Pareto Front Size Over Generations")
    plt.grid(True); plt.legend(); 
    title = plt.gca().get_title()
    save_title = title.replace(' ', '_') + '.png'
    save_path = os.path.join(def_save, save_title)
    plt.savefig(save_path)
    plt.show()

    # --- 打印关键信息 ---
    print("\n===== Key Metrics =====")
    print(f"Initial best distance: {min_vals[0,0]:.2f}, worst: {max_vals[0,0]:.2f}")
    print(f"Final   best distance: {min_vals[-1,0]:.2f}, worst: {max_vals[-1,0]:.2f}")
    impr = (min_vals[0,0] - min_vals[-1,0]) / min_vals[0,0] * 100
    print(f"Improvement in best distance: {impr:.1f}%\n")

    print(f"Initial best demand diff: {min_vals[0,1]:.2f}, worst: {max_vals[0,1]:.2f}")
    print(f"Final   best demand diff: {min_vals[-1,1]:.2f}, worst: {max_vals[-1,1]:.2f}")
    impr2 = (min_vals[0,1] - min_vals[-1,1]) / min_vals[0,1] * 100
    print(f"Improvement in best demand diff: {impr2:.1f}%\n")

    # 初代指标
    print("=== Initial generation metrics ===")
    print(f"Hypervolume          : {HV[0]:.2f}")
    print(f"Spacing              : {SP[0]:.2f}")
    print(f"Range Spread         : {RS[0]:.2f}")
    print(f"Population Diversity : {DV[0]:.2f}")
    print(f"Pareto Front Size    : {PS[0]}")

    print("\n=== Final generation metrics ===")
    print(f"Final hypervolume    : {HV[-1]:.2f}")
    print(f"Final spacing        : {SP[-1]:.2f}")
    print(f"Final spread         : {RS[-1]:.2f}")
    print(f"Final diversity      : {DV[-1]:.2f}")
    print(f"Pareto Front Size    : {PS[-1]}")

    print("========================\n")

    # --- 聚类选代表解：示例 K-Means ---
    # 这段代码放在你完成 metrics 计算并且绘制图表之前，用于从最终 Pareto 前沿中选取代表性解
    # 提取最终前沿的适应度值
    final_front = tools.sortNondominated(populations[-1], k=len(populations[-1]), first_front_only=True)[0]
    front_vals_full = np.array([ind.fitness.values for ind in final_front])
    best_dist_idx = np.argmin(front_vals_full[:,0])
    best_diff_idx = np.argmin(front_vals_full[:,1])

    # 3) 找到最佳折中解：最接近理想点 (min_f1, min_f2)
    # 先计算理想点
    f1_min, f2_min = front_vals_full.min(axis=0)
    f1_max, f2_max = front_vals_full.max(axis=0)

    norm = (front_vals_full - np.array([f1_min, f2_min])) / np.array([f1_max-f1_min, f2_max-f2_min])
    best_balance_idx = np.argmin(np.linalg.norm(norm, axis=1))
    rep_inds = []
    for idx in (best_dist_idx, best_diff_idx, best_balance_idx):
        if idx not in rep_inds:
            rep_inds.append(idx)

    rep_vals = front_vals_full[rep_inds]

    print("=== Three Representative Solutions ===")
    labels = ["Best Min Total Distance", "Best Min Demand Diff", "Best Balanced"]
    for i, idx in enumerate(rep_inds):
        ind = final_front[idx]
        f1, f2 = ind.fitness.values
        print(f"{labels[i]}: {f1:.2f}, {f2:.0f}")
        routes = decode(ind)    # 使用你之前定义的 decode() 函数
        drawMap(city_list, routes, title=f"Representative Solution for {labels[i]}")

    # 画代表解
    plt.figure()
    plt.scatter(front_vals_full[:, 0], front_vals_full[:, 1], c='gray', alpha=0.5, label="Pareto Front")
    plt.scatter(np.array(rep_vals)[:, 0], np.array(rep_vals)[:, 1], c='red', label="Representative Solutions")
    plt.xlabel("Total Distance")
    plt.ylabel("Demand Difference")
    plt.title("Representative Solutions on Pareto Front")
    plt.legend()
    plt.grid(True)
    title = plt.gca().get_title()
    save_title = title.replace(' ', '_') + '.png'
    save_path = os.path.join(def_save, save_title)
    plt.savefig(save_path)
    plt.show()

    sys.stdout = sys.__stdout__
    text_path = os.path.join(def_save, 'metrics.txt')
    with open(text_path, 'w', encoding='utf-8') as f:
        f.write(buffer.getvalue())

    print("---saving---")
    print("Metrics & Graphs saved successfully!")


if __name__ == "__main__":
    main()
