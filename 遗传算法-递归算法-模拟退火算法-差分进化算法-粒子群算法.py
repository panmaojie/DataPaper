import copy
import math
import random
import pandas as pd
import numpy as np
import networkx as nx
import sko
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


class VRP:
    def __init__(self):
        self.demand = data_demand
        self.n_cluster = n_cluster
        self.max_capacity = max_capacity
        self.data = pd.DataFrame()
        self.triplets = []
        self.sublists = []
        self.subdemands_sum = []
        self.pos = {}
        self.twocenters = []
        self.mtsp = []
        self.mtsp_length = []
        self.mtsp_length_sum = 0
        self.mtsp_edges = []

    def convert_to_triplets(self, data):
        triplets = []
        for i in data.index:
            for j in data.columns:
                if not np.isnan(data.iloc[i - 1, j - 1]):
                    triplets.append([i, j, data.iloc[i - 1, j - 1]])
        self.data = data
        self.triplets = triplets
        return self

    def demands_sum(self, sublist=None):
        if sublist == None:
            subdemands = [[] for _ in range(self.n_cluster)]
            subdemands_sum = []
            for i in range(len(self.sublists)):
                for j in range(len(self.sublists[i])):
                    subdemands[i].append(self.demand.iloc[0, self.sublists[i][j] - 1])
                subdemands_sum.append(sum(subdemands[i]))
            self.subdemands_sum = subdemands_sum
            return self.subdemands_sum
        else:
            subdemand = [[] for _ in range(self.n_cluster)]
            subdemand_sum = []
            for i in range(len(sublist)):
                for j in range(len(sublist[i])):
                    subdemand[i].append(self.demand.iloc[0, sublist[i][j] - 1])
                subdemand_sum.append(sum(subdemand[i]))
            return subdemand_sum

    def mileage(self, sublists):
        self.choose_two_centers(sublists)
        return self.mtsp_length

    def isneighboring(self, parent, p1=None, p2=None, i=None, j=None):
        if p1 != None and p2 != None and i != None and j != None:
            temp = parent[p1][i]
            parent[p1][i] = parent[p2][j]
            parent[p2][j] = temp
        tmp_triplets = [[] for _ in range(self.n_cluster)]
        for a in range(len(parent)):
            for b in range(len(parent[a]) - 1):
                for c in range(b + 1, len(parent[a])):
                    for d in range(len(self.triplets)):
                        if parent[a][b] == self.triplets[d][0] and parent[a][c] == self.triplets[d][1]:
                            tmp_triplets[a].append(self.triplets[d])

        for t in range(len(tmp_triplets)):
            if not tmp_triplets[t]:
                return False
        for l in range(len(parent)):
            tmpG = nx.Graph()
            tmpG.add_nodes_from(list(range(1, len(self.data) + 1)))
            tmpG.add_weighted_edges_from(tmp_triplets[l])
            for m in range(len(parent[l]) - 1):
                for n in range(m + 1, len(parent[l])):
                    if nx.has_path(tmpG, parent[l][m], parent[l][n]) == False:
                        return False
        self.sublists = parent
        return True

    def generate_individual(self):
        sublists = [[] for _ in range(n_cluster)]
        first_node = random.sample(range(1, len(self.data) + 1), n_cluster)
        left_nodes = [int(i) for i in range(1, len(self.data) + 1) if i not in first_node]
        for i in range(len(first_node)):
            sublists[i].append(first_node[i])

        G = nx.Graph()
        G.add_weighted_edges_from(self.triplets)
        for i in range(len(left_nodes)):
            temp_dijklength = [[] for _ in range(len(sublists))]
            for j in range(len(sublists)):
                for k in range(len(sublists[j])):
                    dijklength = nx.dijkstra_path_length(G, source=sublists[j][k], target=left_nodes[i])
                    temp_dijklength[j].append(dijklength)
            min_dijk = min([min(m) for m in temp_dijklength])
            for l in range(len(temp_dijklength)):
                if min_dijk in temp_dijklength[l]:
                    sublists[l].append(left_nodes[i])
                    break
        self.sublists = sublists
        self.demands_sum()
        if max(self.subdemands_sum) > self.max_capacity or self.isneighboring(self.sublists) == False:
            return self.generate_individual()
        else:
            return self

    def fitness(self, sublists=None):
        if sublists == None:
            twocenters, mtsp_edges = self.choose_two_centers(self.sublists)
            mtsp_edges, sum_mtsp_length = self.solution(self.sublists, twocenters)
            return sum_mtsp_length
        else:
            twocenters, mtsp_edges = self.choose_two_centers(sublists)
            mtsp_edges, sum_mtsp_length = self.solution(sublists, twocenters)
            return sum_mtsp_length

    def clone(self, fitness, population_fitness, clone_rate):
        if fitness < sorted(population_fitness, reverse=False)[int(len(population_fitness) * clone_rate)]:
            return True
        else:
            return False

    def crossover(self, parent, crossover_rate):
        p1, p2 = np.random.choice(list(range(len(parent))), size=2, replace=False)
        for i in range(len(parent[p1])):
            for j in range(len(parent[p2])):
                for k in range(len(self.triplets)):
                    if parent[p1][i] == self.triplets[k][0] and parent[p2][j] == self.triplets[k][1] and self.isneighboring(parent, p1, p2, i, j) == True:
                        if random.random() < crossover_rate:
                            temp = parent[p1][i]
                            parent[p1][i] = parent[p2][j]
                            parent[p2][j] = temp
                            return parent
        return parent

    def mutate(self, parent, mutation_rate):
        p1, p2 = np.random.choice(list(range(len(parent))), size=2, replace=False)
        flag = False
        for i in range(len(parent[p1])):
            for j in range(len(parent[p2])):
                for k in range(len(self.triplets)):
                    if parent[p1][i] == self.triplets[k][0] and parent[p2][j] == self.triplets[k][1] and self.isneighboring(parent, p1, p2, i, j) == True:
                        if random.random() < mutation_rate and len(parent[p1]) > 1 and not flag:
                            elem_move = parent[p1].pop(i)
                            parent[p2].append(elem_move)
                            flag = True
            break
        return parent

    def solution(self, cluster_node, twocenters):
        cluster_node2 = copy.deepcopy(cluster_node)
        far_index = []
        if twocenters[0] not in cluster_node2[0]:
            cluster_node2[0].append(twocenters[0])
            far_index.append(0)
        else:
            cluster_node2[1].append(twocenters[0])
            far_index.append(1)
        if twocenters[1] not in cluster_node2[2]:
            cluster_node2[2].append(twocenters[1])
            far_index.append(2)
        else:
            cluster_node2[3].append(twocenters[1])
            far_index.append(3)

        temp_G = nx.Graph()
        temp_G.add_weighted_edges_from(self.triplets)
        far = []
        for c in cluster_node[:2]:
            if twocenters[0] not in c:
                far.append(c)
        for c in cluster_node[2:]:
            if twocenters[1] not in c:
                far.append(c)

        temp_dijkstra = []
        temp_dijkstra_length = []

        for i in range(len(far[0])):
            temp_dijk = nx.dijkstra_path(temp_G, source=twocenters[0], target=far[0][i])
            temp_dijk_length = nx.dijkstra_path_length(temp_G, source=twocenters[0], target=far[0][i])
            temp_dijkstra.append(temp_dijk)
            temp_dijkstra_length.append(temp_dijk_length)
        min_len_index = temp_dijkstra_length.index(min(temp_dijkstra_length))
        for i in temp_dijkstra[min_len_index]:
            if i not in cluster_node2[far_index[0]]:
                cluster_node2[far_index[0]].append(i)

        temp_dijkstra2 = []
        temp_dijkstra_length2 = []

        for i in range(len(far[1])):
            temp_dijk2 = nx.dijkstra_path(temp_G, source=twocenters[1], target=far[1][i])
            temp_dijk_length2 = nx.dijkstra_path_length(temp_G, source=twocenters[1], target=far[1][i])
            temp_dijkstra2.append(temp_dijk2)
            temp_dijkstra_length2.append(temp_dijk_length2)
        min_len_index2 = temp_dijkstra_length2.index(min(temp_dijkstra_length2))
        for i in temp_dijkstra2[min_len_index2]:
            if i not in cluster_node2[far_index[1]]:
                cluster_node2[far_index[1]].append(i)

        edges = [[] for _ in range(len(cluster_node2))]
        for i in range(len(cluster_node2)):
            for j in range(0, len(cluster_node2[i]) - 1):
                for k in range(1, len(cluster_node2[i])):
                    if ~np.isnan(self.data.iloc[cluster_node2[i][j] - 1, cluster_node2[i][k] - 1]):
                        edges[i].append([cluster_node2[i][j], cluster_node2[i][k], self.data.iloc[cluster_node2[i][j] - 1, cluster_node2[i][k] - 1]])

        mtsp = []
        for e in range(len(edges)):
            mG = nx.Graph()
            mG.add_weighted_edges_from(edges[e])
            mtsp.append(nx.algorithms.approximation.traveling_salesman_problem(mG, weight='weight'))

        temp_mtsp = [[] for _ in range(len(cluster_node2))]
        for i in range(len(mtsp)):
            if i < 2:
                for j in range(len(mtsp[i])):
                    if mtsp[i][j] == twocenters[0]:
                        temp_mtsp[i].append(mtsp[i][j:-1])
                        temp_mtsp[i].append(mtsp[i][:j + 1])
                        break
            else:
                for j in range(len(mtsp[i])):
                    if mtsp[i][j] == twocenters[1]:
                        temp_mtsp[i].append(mtsp[i][j:-1])
                        temp_mtsp[i].append(mtsp[i][:j + 1])
                        break
        for i in range(len(temp_mtsp)):
            temp_mtsp[i] = sum(temp_mtsp[i], [])
        mtsp = temp_mtsp
        mtsp_edges = [[] for _ in range(len(cluster_node2))]
        for e in range(len(mtsp)):
            for i in range(0, len(mtsp[e]) - 1):
                mtsp_edges[e].append([mtsp[e][i], mtsp[e][i + 1]])

        mtsp_length = [0 for _ in range(len(cluster_node2))]
        for i in range(len(mtsp_edges)):
            for j in range(len(mtsp_edges[i])):
                for k in range(len(self.triplets)):
                    if self.triplets[k][0] == mtsp_edges[i][j][0] and self.triplets[k][1] == mtsp_edges[i][j][1]:
                        mtsp_length[i] += self.triplets[k][2]

        self.twocenters = twocenters
        self.mtsp = mtsp
        self.mtsp_length = mtsp_length
        self.mtsp_length_sum = sum(mtsp_length)
        return mtsp_edges, sum(mtsp_length)

    def choose_two_centers(self, cluster_node):
        temp_cluster_node = []
        temp_cluster_node.append(cluster_node[:2])
        temp_cluster_node.append(cluster_node[2:])
        for i in range(len(temp_cluster_node)):
            temp_cluster_node[i] = sum(temp_cluster_node[i], [])

        twocenters = [0, 0]
        mtsp_length_sum = 9999
        mtsp_edges = 0
        for i in temp_cluster_node[0]:
            for j in temp_cluster_node[1]:
                temp_mtsp_edges, mtsp_len_sum = self.solution(cluster_node, [i, j])
                if mtsp_len_sum < mtsp_length_sum:
                    mtsp_length_sum = mtsp_len_sum
                    twocenters[0] = i
                    twocenters[1] = j
                    mtsp_edges = temp_mtsp_edges
        mtsp_edges, mtsp_length_sum = self.solution(cluster_node, twocenters)
        self.mtsp_edges = mtsp_edges
        self.mtsp_length_sum = mtsp_length_sum
        return twocenters, mtsp_edges

    def graph(self, cluster_node, best_fitness, title):
        while True:
            if len(best_fitness) < 200:
                best_fitness.append(best_fitness[-1])
            else:
                break

        plt.figure(figsize=(6, 6))
        plt.plot(best_fitness)
        plt.title(title)
        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plt.savefig(fname='images/Fitness Function Curve Diagram - '+title+'.png')

        G = nx.Graph()
        G.add_weighted_edges_from(self.triplets)
        labels = nx.get_edge_attributes(G, 'weight')
        node_colors = ['#00bfff', '#00ff00', '#ff6666', '#ccaaff']
        edge_colors = ['blue', 'green', 'red', 'purple']

        plt.figure(figsize=(6, 6))
        nx.draw(G, self.pos, with_labels=True)
        nx.draw_networkx_nodes(G, self.pos, node_color='yellow', edgecolors='red')
        nx.draw_networkx_edge_labels(G, self.pos, edge_labels=labels, font_color='purple', font_size=10)
        plt.savefig(fname='images/Original Node Distribution Map - '+title+'.png')

        plt.figure(figsize=(6, 6))
        nx.draw(G, self.pos, with_labels=True)
        for i in range(len(cluster_node)):
            nx.draw_networkx_nodes(G, self.pos, nodelist=cluster_node[i], node_color=node_colors[i], edgecolors='black')
        nx.draw_networkx_edge_labels(G, self.pos, edge_labels=labels, font_color='purple', font_size=10)
        plt.savefig(fname='images/Node Clustering Distribution Map - ' + title + '.png')

        plt.figure(figsize=(6, 6))
        nx.draw(G, self.pos, with_labels=True)
        for i in range(len(cluster_node)):
            nx.draw_networkx_nodes(G, self.pos, nodelist=cluster_node[i], node_color=node_colors[i], edgecolors='black')
        nx.draw_networkx_nodes(G, self.pos, nodelist=self.twocenters, node_color='yellow', edgecolors='#1f77b4', node_shape='s', node_size=600)
        for m in range(len(self.mtsp_edges)):
            nx.draw_networkx_edges(G, self.pos, edgelist=self.mtsp_edges[m], edge_color=edge_colors[m], width=3, arrows=True, arrowstyle='->', arrowsize=15)
        nx.draw_networkx_edge_labels(G, self.pos, edge_labels=labels, font_color='purple', font_size=10)
        plt.savefig(fname='images/Node Route Planning Distribution Map - '+title+'.png')

    def genetic_algorithm(self, data, demand, n_cluster, max_capacity, pos):
        self.pos = pos
        self.convert_to_triplets(data)
        population_size = 20
        clone_rate = 0.1
        crossover_rate = 0.5
        mutation_rate = 0.5
        max_generations = 30

        population = []
        demands_sum = []
        mileage = []
        population_fitness = []

        print('初始化种群……')
        # random.seed(6)
        for i in range(population_size):
            self.generate_individual()
            population.append(self.sublists)
            demands_sum.append(self.subdemands_sum)
            mileage.append(self.mtsp_length)
            population_fitness.append(self.fitness())

        best_populations = []
        best_demands_sum = []
        best_mileage = []
        best_fitness = []

        try:
            for i in range(max_generations):
                child_populations = []
                child_demands_sum = []
                child_mileage = []
                child_fitness = []
                for j in range(population_size):
                    print('第', i + 1, '代，第', j + 1, '个种群个体')
                    if best_populations != [] and len(population) == len(child_populations) + 1:
                        child_populations.append(best_populations[-1])
                        child_demands_sum.append(best_demands_sum[-1])
                        child_mileage.append(best_mileage[-1])
                        child_fitness.append(best_fitness[-1])
                    elif self.clone(population_fitness[j], population_fitness, clone_rate * (population_size + j) / population_size) == True and self.isneighboring(population[j]) == True:
                        child_populations.append(population[j])
                        child_demands_sum.append(self.demands_sum(population[j]))
                        child_mileage.append(self.mileage(population[j]))
                        child_fitness.append(self.fitness(population[j]))
                    else:
                        pop_cro = self.crossover(population[j], crossover_rate * population_size / (population_size + j))
                        if max(self.demands_sum(pop_cro)) <= self.max_capacity and self.isneighboring(pop_cro) == True:
                            child_populations.append(pop_cro)
                            child_demands_sum.append(self.demands_sum(pop_cro))
                            child_mileage.append(self.mileage(pop_cro))
                            child_fitness.append(self.fitness(pop_cro))
                        else:
                            pop_mut = self.mutate(pop_cro, mutation_rate * population_size / (population_size + j))
                            if max(self.demands_sum(pop_mut)) <= self.max_capacity and self.isneighboring(pop_mut) == True:
                                child_populations.append(pop_mut)
                                child_demands_sum.append(self.demands_sum(pop_mut))
                                child_mileage.append(self.mileage(pop_mut))
                                child_fitness.append(self.fitness(pop_mut))
                            else:
                                self.generate_individual()
                                child_populations.append(self.sublists)
                                child_demands_sum.append(self.subdemands_sum)
                                child_mileage.append(self.mtsp_length)
                                child_fitness.append(self.fitness())

                print(child_fitness)
                best_populations.append(child_populations[child_fitness.index(min(child_fitness))])
                best_demands_sum.append(child_demands_sum[child_fitness.index(min(child_fitness))])
                best_mileage.append(child_mileage[child_fitness.index(min(child_fitness))])
                best_fitness.append(min(child_fitness))

                population = child_populations
                demands_sum = child_demands_sum
                mileage = child_mileage
                population_fitness = child_fitness

        except KeyboardInterrupt or KeyError or TypeError:
            pass

        for i in range(len(best_populations)):
            print(best_populations[i], best_demands_sum[i], best_mileage[i], best_fitness[i])

        self.choose_two_centers(best_populations[-1])
        print('物资中心：', self.twocenters, '\n各回路节点：', best_populations[-1], '\n各回路路径：', self.mtsp, '\n各回路载重量：', best_demands_sum[-1], '\n各回路里程：', best_mileage[-1], '总里程：', best_fitness[-1])
        self.graph(best_populations[-1], best_fitness, title='Genetic Algorithm')
        return best_fitness

    def recursive(self, iterations, opportunities, optimal_solution, optimal_demands_sum, optimal_mileage, optimal_fitness, total_fitness):
        if iterations == 0 or opportunities == 0:
            return optimal_solution, optimal_demands_sum, optimal_mileage, optimal_fitness, total_fitness
        else:
            if iterations == 20:
                solution = None
                demands_sum = None
                mileage = None
                fitness = 9999
            else:
                solution = optimal_solution
                demands_sum = optimal_demands_sum
                mileage = optimal_mileage
                fitness = optimal_fitness
            while True:
                self.generate_individual()
                opportunities -= 1
                solution = self.sublists
                demands_sum = self.demands_sum(solution)
                mileage = self.mileage(solution)
                tmp_fitness = self.fitness(solution)
                if tmp_fitness <= fitness:
                    optimal_solution = solution
                    optimal_demands_sum = demands_sum
                    optimal_mileage = mileage
                    optimal_fitness = tmp_fitness
                    total_fitness.append(optimal_fitness)
                    break
                elif opportunities == 0:
                    total_fitness.append(optimal_fitness)
                    break
                else:
                    continue
            print(iterations, optimal_solution, optimal_demands_sum, optimal_mileage, optimal_fitness)
            iterations -= 1
            return self.recursive(iterations, opportunities, optimal_solution, optimal_demands_sum, optimal_mileage, optimal_fitness, total_fitness)

    def recursive_algorithm(self, data, demand, n_cluster, max_capacity, pos):
        self.pos = pos
        iterations = 20
        opportunities = 30
        self.convert_to_triplets(data)
        optimal_solution = 0
        optimal_demands_sum = 0
        optimal_mileage = 0
        optimal_fitness = 0
        total_fitness = []

        try:
            optimal_solution, optimal_demands_sum, optimal_mileage, optimal_fitness, total_fitness = self.recursive(iterations, opportunities, optimal_solution, optimal_demands_sum, optimal_mileage, optimal_fitness, total_fitness)
            self.choose_two_centers(optimal_solution)
        except KeyboardInterrupt or KeyError or TypeError:
            pass
        print('物资中心：', self.twocenters, '\n各回路节点：', optimal_solution, '\n各回路路径：', self.mtsp, '\n各回路载重量：', optimal_demands_sum, '\n各回路里程：', optimal_mileage, '总里程：', optimal_fitness)
        self.graph(optimal_solution, total_fitness, title='Recursive Algorithm')
        return total_fitness

    def metropolis(self, current_fitness, new_fitness, temperature_max):
        if new_fitness < current_fitness:
            return True
        else:
            return math.exp((current_fitness - new_fitness) / temperature_max)

    def simulated_annealing_algorithm(self, data, demand, n_cluster, max_capacity, pos):
        self.pos = pos
        self.convert_to_triplets(data)
        self.generate_individual()

        current_solution = self.sublists
        current_demands_sum = self.demands_sum(current_solution)
        current_mileage = self.mileage(current_solution)
        current_fitness = self.fitness(current_solution)
        total_fitness = []

        best_solution = current_solution
        best_demands_sum = current_demands_sum
        best_mileage = current_mileage
        best_fitness = current_fitness

        temperature_max = 5000
        temperature_min = 0.001
        annealing_rate = 0.5
        iterations = 1

        while temperature_max > temperature_min:
            self.generate_individual()
            new_solution = self.sublists
            new_demands_sum = self.demands_sum(new_solution)
            new_mileage = self.mileage(new_solution)
            new_fitness = self.fitness(new_solution)

            metropolis = self.metropolis(current_fitness, new_fitness, temperature_max)

            if random.random() < metropolis:
                current_solution = new_solution
                current_demands_sum = new_demands_sum
                current_mileage = new_mileage
                current_fitness = new_fitness

            if current_fitness < best_fitness:
                best_solution = current_solution
                best_demands_sum = new_demands_sum
                best_mileage = new_mileage
                best_fitness = new_fitness

            print(iterations, best_solution, best_demands_sum, best_mileage, best_fitness)
            total_fitness.append(best_fitness)
            temperature_max *= annealing_rate
            iterations += 1

        self.choose_two_centers(best_solution)
        print('物资中心：', self.twocenters, '\n各回路节点：', best_solution, '\n各回路路径：', self.mtsp, '\n各回路载重量：', best_demands_sum, '\n各回路里程：', best_mileage, '总里程：', best_fitness)
        self.graph(best_solution, total_fitness, title='Simulated Annealing Algorithm')
        return total_fitness

    def differential_evolution_algorithm(self, data, demand, n_cluster, max_capacity, pos):
        self.pos = pos
        self.convert_to_triplets(data)

        population_size = 100
        iterations_max = 50
        population = []
        demands_sum = []
        mileage = []
        population_fitness = []

        print('初始化种群……')
        for i in range(population_size):
            print('第', i + 1, '个种群个体')
            self.generate_individual()
            population.append(self.sublists)
            demands_sum.append(self.subdemands_sum)
            mileage.append(self.mtsp_length)
            population_fitness.append(self.fitness())

        def cal_fitness(i):
            i = int(i)
            return population_fitness[i]

        dea = sko.DE.DE(cal_fitness, n_dim=1, size_pop=population_size, max_iter=iterations_max, lb=[0], ub=[population_size-1])
        dea.run()

        best_solution = population[population_fitness.index(dea.best_y)]
        best_demands_sum = demands_sum[population_fitness.index(dea.best_y)]
        best_mileage = mileage[population_fitness.index(dea.best_y)]
        best_fitness = dea.best_y
        total_fitness = dea.generation_best_Y
        print(total_fitness)
        self.choose_two_centers(best_solution)
        print('物资中心：', self.twocenters, '\n各回路节点：', best_solution, '\n各回路路径：', self.mtsp, '\n各回路载重量：', best_demands_sum, '\n各回路里程：', best_mileage, '总里程：', best_fitness)
        self.graph(best_solution, total_fitness, title='Differential Evolution Algorithm')
        return total_fitness

    def particle_swarm_optimization(self, data, demand, n_cluster, max_capacity, pos):
        self.pos = pos
        self.convert_to_triplets(data)

        population_size = 100
        iterations_max = 50
        population = []
        demands_sum = []
        mileage = []
        population_fitness = []

        print('初始化种群……')
        for i in range(population_size):
            print('第', i + 1, '个种群个体')
            self.generate_individual()
            population.append(self.sublists)
            demands_sum.append(self.subdemands_sum)
            mileage.append(self.mtsp_length)
            population_fitness.append(self.fitness())

        def cal_fitness(i):
            i = int(i)
            return population_fitness[i]

        pso = sko.PSO.PSO(cal_fitness, n_dim=1, pop=population_size, max_iter=iterations_max, lb=[0], ub=[population_size-1], w=0.8, c1=0.5, c2=0.5)
        pso.run()

        best_solution = population[population_fitness.index(pso.best_y)]
        best_demands_sum = demands_sum[population_fitness.index(pso.best_y)]
        best_mileage = mileage[population_fitness.index(pso.best_y)]
        best_fitness = pso.best_y
        total_fitness = pso.gbest_y_hist
        self.choose_two_centers(best_solution)
        print('物资中心：', self.twocenters, '\n各回路节点：', best_solution, '\n各回路路径：', self.mtsp, '\n各回路载重量：', best_demands_sum, '\n各回路里程：', best_mileage, '总里程：', best_fitness)
        self.graph(best_solution, total_fitness, title='Particle Swarm Optimization')
        return total_fitness


if __name__ == '__main__':
    data_car = pd.read_excel('数据.xlsx', '各地点之间距离(车辆)', index_col=0)
    data_air = pd.read_excel('数据.xlsx', '各地点之间距离(无人机)', index_col=0)
    data_demand = pd.read_excel('数据.xlsx', '各地点平均日物资需求量', index_col=0)
    n_cluster = 4
    max_capacity = 500

    pos = {1: (35, 440), 2: (140, 480), 3: (345, 490), 4: (390, 440), 5: (180, 420), 6: (320, 400), 7: (115, 365), 8: (200, 355), 9: (270, 345), 10: (395, 355),
           11: (80, 290), 12: (190, 260), 13: (285, 255), 14: (380, 290), 15: (120, 210), 16: (220, 210), 17: (390, 225), 18: (30, 200), 19: (95, 150), 20: (255, 140),
           21: (325, 190), 22: (405, 165), 23: (325, 130), 24: (85, 85), 25: (185, 115), 26: (280, 75), 27: (420, 85), 28: (355, 35), 29: (160, 35), 30: (255, 30)}

    ga = VRP().genetic_algorithm(data_air, data_demand, n_cluster, max_capacity, pos)  # 遗传算法：迭代慢（运行半小时或者更久），自己写的

    ra = VRP().recursive_algorithm(data_air, data_demand, n_cluster, max_capacity, pos)  # 递归算法：自己写的

    saa = VRP().simulated_annealing_algorithm(data_air, data_demand, n_cluster, max_capacity, pos)  # 模拟退火算法：自己写的

    dea = VRP().differential_evolution_algorithm(data_air, data_demand, n_cluster, max_capacity, pos)  # 差分进化算法：套用第三方库

    pso = VRP().particle_swarm_optimization(data_air, data_demand, n_cluster, max_capacity, pos)  # 粒子群算法：套用第三方库

    for i in [ga, ra, saa, dea, pso]:
        while True:
            if len(i) < 200:
                i.append(i[-1])
            else:
                break
    list_total=[]
    for i in [ga, ra, saa, dea, pso]:
        i=pd.DataFrame(i)
        list_total.append(i)
    fitness_total = pd.concat(list_total, axis=1)
    fitness_total.columns = ['Genetic Algorithm', 'Recursive Algorithm', 'Simulated Annealing Algorithm', 'Differential Evolution Algorithm', 'Particle Swarm Optimization']
    print(fitness_total)
    fitness_total.to_excel('各算法适应度对比.xlsx', index=None, columns=['Genetic Algorithm', 'Recursive Algorithm', 'Simulated Annealing Algorithm', 'Differential Evolution Algorithm', 'Particle Swarm Optimization'])

    # fitness_total=pd.read_excel('各算法适应度对比.xlsx',sheet_name='Sheet1')

    # ga=fitness_total.iloc[:,0]
    # ra = fitness_total.iloc[:, 1]
    # saa = fitness_total.iloc[:, 2]
    # dea = fitness_total.iloc[:, 3]
    # pso = fitness_total.iloc[:, 4]

    fig = plt.figure(figsize=(6, 6))
    fig.patch.set_facecolor('#75bbfd')
    fig.patch.set_alpha(0.3)
    ax = plt.axes()
    ax.set_facecolor('white')
    ax.patch.set_alpha(1)
    plt.plot(ga, '--*', color='#ff474c',markevery=25,markersize=6)
    plt.plot(ra, '--^', color='#0504aa',markevery=25,markersize=6)
    plt.plot(saa, '--D', color='#06c2ac',markevery=25,markersize=6)
    plt.plot(dea, '--o', color='#fdff52',markevery=25,markersize=6)
    plt.plot(pso, '--P', color='#fb5ffc',markevery=25,markersize=6)
    plt.xlabel('Iterations')
    plt.ylabel('Fitness')
    plt.legend(['Genetic Algorithm', 'Recursive Algorithm', 'Simulated Annealing Algorithm', 'Differential Evolution Algorithm', 'Particle Swarm Optimization'])
    plt.savefig(fname='images/Fitness Function Curve Diagram.png')
    plt.show()

