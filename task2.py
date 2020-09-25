from pyspark import SparkContext, SparkConf
import sys
import csv
import os
import time
from pyspark.sql import SparkSession
from itertools import combinations
from operator import add
import random

def girwan_newman(vertex, vertices_count, adjacency_list):
    visited = []
    current_level, next_level = [vertex], []
    visited.append(vertex)
    levels_list = []
    vertices_in_current_level = []
    while current_level:
        current_vertex = current_level.pop(0)
        for adjacent_vertex in adjacency_list[current_vertex]:
            if adjacent_vertex not in visited:
                visited.append(adjacent_vertex)
                next_level.append(adjacent_vertex)
        vertices_in_current_level.append(current_vertex)

        if not current_level:
            current_level = next_level
            next_level = []
            levels_list.append(vertices_in_current_level)
            vertices_in_current_level = []

    vertex_weights = {}
    vertex_weights[vertex] = 1
    levels = len(levels_list)
    if levels >= 2:
        for node in levels_list[1]:
            vertex_weights[node] = 1

    for cur_level in range(2, levels):
        current_level_nodes = levels_list[cur_level]
        parent_level_nodes = levels_list[cur_level - 1]
        for vertex in current_level_nodes:
            parents = set(adjacency_list[vertex]).intersection(set(parent_level_nodes))
            vertex_weights[vertex] = sum([vertex_weights[parent] for parent in parents])
    
    reverse_level_list = list(reversed(levels_list))
    edge_weights_dict = {}
    for i, cur_level in enumerate(reverse_level_list[:-1]):
        for vertex in cur_level:    
            split = 1
            if i != 0:
                children = set(adjacency_list[vertex]).intersection(set(reverse_level_list[i - 1]))
                split = 1 + sum([edge_weights_dict[(min(vertex, child), max(vertex, child))] for child in children])
            parents = set(adjacency_list[vertex]).intersection(set(reverse_level_list[i + 1]))
            total_parent_weight = sum([vertex_weights[parent] for parent in parents])
            for parent in parents:
                edge_weights_dict[(min(vertex, parent), max(vertex, parent))] = (float(vertex_weights[parent])/float(total_parent_weight)) * float(split)

    return edge_weights_dict


def calculateBetweenness(vertices, vertices_count, adjacency_list):
    total_edge_weights = dict()
    for vertex in vertices:
        current_edge_weights = girwan_newman(vertex, vertices_count, adjacency_list)
        for edge in current_edge_weights:
            total_edge_weights[edge] = total_edge_weights.get(edge, 0) + current_edge_weights[edge]
    yield total_edge_weights.items()

def get_initial_modularity(vertex_degree_dict, vertices_set):
    modularity_map = {}
    A, q, modularity, max_modularity = 0,0,0.0,0.0
    for pair in combinations(vertices_set, 2):
        if max(pair[0],  pair[1]) in adjacency_list[min(pair[0],  pair[1])]:
            A = 1
        else:
            A = 0
        q = (A - ((0.5 * vertex_degree_dict[min(pair[0],  pair[1])] * vertex_degree_dict[max(pair[0],  pair[1])]) / edges_count))
        modularity_map[(min(pair[0],  pair[1]), max(pair[0],  pair[1]))] = q
        modularity += q

    max_modularity = float(1 / (float(2 * edges_count))) * modularity
    return modularity_map, max_modularity

def get_communities(adjacency_list, vertices_set):
    communities_list = list() 
    already_seen = list()  
    current_community_nodes_set = set() 
    visited = set()  

    random_root = list(vertices_set)[random.randint(0, len(vertices_set) - 1)]
    current_community_nodes_set.add(random_root)
    already_seen.append(random_root)
    while len(visited) != len(vertices_set):
        while len(already_seen) > 0:
            parent_node = already_seen.pop(0)
            current_community_nodes_set.add(parent_node)
            visited.add(parent_node)
            for children in adjacency_list[parent_node]:
                if children not in visited:
                    current_community_nodes_set.add(children)
                    already_seen.append(children)
                    visited.add(children)

        communities_list.append(sorted(current_community_nodes_set))
        current_community_nodes_set = set()
        if len(vertices_set) > len(visited):
            already_seen.append(set(vertices_set).difference(visited).pop())

    return communities_list

if __name__ == "__main__":
    start_time = time.time()
    input_file_path = sys.argv[2]
    filter_threshold = sys.argv[1]
    betweeness_output_file_path = sys.argv[3]
    communities_ouptut_file_path = sys.argv[4]

    #input_file_path = "./Data/ub_sample_data.csv"

    conf = SparkConf().set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    sparkSession = SparkSession(sc)
    sc.setLogLevel("ERROR")

    user_business_map = sc.textFile(input_file_path).filter(lambda line: line.split(",")[0]!="user_id")\
        .map(lambda line : (line.split(",")[0], line.split(",")[1]))\
            .groupByKey().mapValues(lambda x: list(set(x))).map(lambda kv:{kv[0]:kv[1]})\
                .flatMap(lambda items: items.items()).collectAsMap()

    user_combinations = list(combinations(list(user_business_map.keys()), 2))

    edges = list()
    vertices_set = set()
    edges_count = 0 
    for pair in user_combinations: 
        if (len(set(user_business_map[pair[0]]).intersection(set(user_business_map[pair[1]]))) >= int(filter_threshold)):
            edges.append(tuple(pair))
            edges.append(tuple((pair[1], pair[0])))
            vertices_set.add(pair[0])
            vertices_set.add(pair[1])
            edges_count+=1

    adjacency_list = sc.parallelize(edges).groupByKey() \
        .mapValues(lambda uidxs: sorted(list(set(uidxs)))).collectAsMap()
    
    betweenness_edges = sc.parallelize(vertices_set)\
        .mapPartitions(lambda vertices : calculateBetweenness(vertices, len(vertices_set), adjacency_list))\
            .flatMap(list).reduceByKey(add).map(lambda x: (x[0], x[1] / 2)).sortBy(lambda x: -x[1])

    betweenness_edges_final_list = betweenness_edges\
        .map(lambda x: (tuple(sorted((x[0][0], x[0][1]))), x[1])).sortByKey().sortBy(lambda x: -x[1])\
            .collect()

    betweenness_length = len(betweenness_edges_final_list)

    with open(betweeness_output_file_path, 'w') as betweenness_file_writer:
        if betweenness_length > 0:
            for i, e in enumerate(betweenness_edges_final_list):
                betweenness_file_writer.write("('" + e[0][0] + "', '" + e[0][1] + "'), " + str(e[1]))
                if i != betweenness_length - 1:
                    betweenness_file_writer.write("\n")
        betweenness_file_writer.close()


    vertex_degree_dict = {}
    for node in vertices_set:
        vertex_degree_dict[node] = len(adjacency_list[node])

    modularity_map, max_modularity = get_initial_modularity(vertex_degree_dict, vertices_set)
    final_communities  = get_communities(adjacency_list, vertices_set)

    #for edgeIndex in range(1, edges_count + 1):
    while True:
        
        highest_betweenness_vertices = betweenness_edges.take(1)[0]
        #print(highest_betweenness_vertices)
        i = highest_betweenness_vertices[0][0]
        j = highest_betweenness_vertices[0][1]

        adjacency_list[i].remove(j)
        adjacency_list[j].remove(i)

        communities  = get_communities(adjacency_list, vertices_set)

        current_modularity = 0.0
        current_communities = []

        for cluster in communities:
            cluster_modularity = 0.0
            for pair in combinations(cluster, 2):
                cluster_modularity += modularity_map[(min(pair[0], pair[1]), max(pair[0], pair[1]))]
            current_communities.append(sorted(cluster))
            current_modularity += cluster_modularity
        current_modularity = float(1 / (float(2 * edges_count))) * current_modularity

        if current_modularity > max_modularity:
            max_modularity = current_modularity
            print(max_modularity)
            finalCommunities = current_communities
        else:
            break

        betweenness_edges = sc.parallelize(vertices_set)\
        .mapPartitions(lambda vertices : calculateBetweenness(vertices, len(vertices_set), adjacency_list))\
            .flatMap(list).reduceByKey(add).map(lambda x: (x[0], x[1] / 2)).sortBy(lambda x: -x[1])

    final_communities = sorted(final_communities, key=lambda item: (len(item), item[0], item[1]))

    with open(communities_ouptut_file_path, 'w') as communities_file_writer:
        for index, community in enumerate(final_communities):
            communities_file_writer.write("'"+community[0]+"'")
            if len(community) > 1:
                for node in community[1:]:
                    communities_file_writer.write(", ")
                    communities_file_writer.write("'"+node+"'")
            if index != len(final_communities) - 1:
                communities_file_writer.write("\n")
        communities_file_writer.close()

    
    print("duration",time.time()- start_time)