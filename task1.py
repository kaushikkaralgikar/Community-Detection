from pyspark import SparkContext, SparkConf
import sys
import csv
import os
import time
from graphframes import GraphFrame
from pyspark.sql import SparkSession
from itertools import combinations

os.environ["PYSPARK_SUBMIT_ARGS"] = ("--packages graphframes:graphframes:0.6.0-spark2.3-s_2.11")

if __name__ == "__main__":
    start_time = time.time()
    input_file_path = sys.argv[1]
    output_file_path = sys.argv[2]
    filter_threshold = sys.argv[3]

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
    vertices = set()

    for pair in user_combinations: 
        if (len(set(user_business_map[pair[0]]).intersection(set(user_business_map[pair[1]]))) >= filter_threshold):
            edges.append(tuple(pair))
            vertices.add(pair[0])
            vertices.add(pair[1])

    vertices_df = sc.parallelize(list(vertices)).map(lambda uid: (uid,)).toDF(['id'])
    edges_df = sc.parallelize(edges).toDF(["src", "dst"])

    graph  = GraphFrame(vertices_df, edges_df)

    communities = graph.labelPropagation(maxIter=5)

    communities_list = communities.rdd.map(lambda id_label: (id_label[1], id_label[0])) \
        .groupByKey().map(lambda ids: sorted(list(ids[1]))) \
        .sortBy(lambda ids: (len(ids), ids)).collect()

    with open(output_file_path, 'w+') as output_file:
        for item in communities_list:
            output_file.writelines(str(item)[1:-1] + "\n")
        output_file.close()