import pickle
import time

import config
from utils.segk.segk import segk
import networkx as nx


def sequence2list(filename):
    graphs = dict()
    with open(filename, 'r') as f:
        for line in f:
            walks = line.strip().split('\t')[:config.max_sequence+1]
            # put message/cascade id into graphs dictionary, value is a list
            graphs[walks[0]] = list()
            for i in range(1, len(walks)):
                nodes = walks[i].split(":")[0]
                time = walks[i].split(":")[1]
                graphs[walks[0]] \
                    .append([[int(x) for x in nodes.split(",")],
                             int(time)])

    return graphs


def read_labels_and_sizes(filename):
    labels = dict()
    sizes = dict()
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split('\t')
            # parts[-1] means the incremental popularity
            labels[parts[0]] = parts[-1]
            # parts[3] means the observed popularity
            sizes[parts[0]] = int(parts[3])
    return labels, sizes


def write_cascade(graphs, labels, sizes, length, filename,
                  weight=False):
    y_data = list()
    size_data = list()
    time_data = list()
    rnn_index = list()
    embedding = list()
    n_cascades = 0
    new_input = list()
    global_input = list()
    for key, graph in graphs.items():
        temp_time = list()
        temp_index = list()
        temp_size = len(graph)
        for walk in graph:
            # save publish time into temp_time list
            temp_time.append(walk[1])
            # save length of walk into temp_index
            temp_index.append(len(walk[0]))
        size_data.append(temp_size)
        time_data.append(temp_time)
        rnn_index.append(temp_index)
        n_cascades += 1

    # padding the embedding
    embedding_size = config.gc_emd_size

    cascade_i = 0
    cascade_size = len(graphs)
    total_time = 0
    nodex_index_list = []
    for key, graph in graphs.items():
        start_time = time.time()
        new_temp = list()
        global_temp = list()
        dg = nx.DiGraph()
        nodes_index = list()
        list_edge = list()
        cascade_embedding = list()
        global_embedding = list()
        times = list()
        t_o = config.observation_time
        for path in graph:
            t = path[1]
            if t >= t_o:
                continue
            nodes = path[0]
            if len(nodes) == 1:
                nodes_index.extend(nodes)
                times.append(1)
                continue
            else:
                nodes_index.extend([nodes[-1]])
            if weight:
                edge = (nodes[-1], nodes[-2], (1 - t / t_o))
                times.append(1 - t / t_o)
            else:
                edge = (nodes[-1], nodes[-2])
            list_edge.append(edge)
        if weight:
            dg.add_weighted_edges_from(list_edge)
        else:
            dg.add_edges_from(list_edge)
        nodes_index_unique = list(set(nodes_index))
        nodes_index_unique.sort(key=nodes_index.index)
        g = dg
        d = 10
        try:
            if len(nodes_index_unique) >= 12:
                chi = segk(nodes_index_unique, list_edge, 2, d, 'shortest_path')
                for node in nodes_index:
                    cascade_embedding.append(chi[nodes_index_unique.index(node)])

                new_temp.extend(cascade_embedding)
                new_input.append(new_temp)

                total_time += time.time() - start_time
                cascade_i += 1
                label = labels[key].split()
                y = int(label[0])
                y_data.append(y)

                if cascade_i % 100 == 0:
                    speed = total_time / cascade_i
                    eta = (cascade_size - cascade_i) * speed
                    print("{}/{}, eta: {:.2f} minutes".format(
                        cascade_i, cascade_size, eta / 60))
        except IndexError:
            pass

    with open(filename, 'wb') as fin:
        pickle.dump((new_input, y_data), fin)


def get_max_size(sizes):
    max_size = 0
    for cascade_id in sizes:
        max_size = max(max_size, sizes[cascade_id])
    return max_size


def get_max_length(graphs):
    """ Get the max length among sequences. """
    max_length = 0
    for cascade_id in graphs:
        # traverse the graphs for max length sequence
        for sequence in graphs[cascade_id]:
            max_length = max(max_length, len(sequence[0]))
    return max_length


if __name__ == "__main__":
    
    time_start = time.time()

    # get the information of nodes/users of cascades
    graphs_train = sequence2list(config.cascade_shortestpath_train)
    graphs_val = sequence2list(config.cascade_shortestpath_validation)
    graphs_test = sequence2list(config.cascade_shortestpath_test)

    # get the information of labels and sizes of cascades
    labels_train, sizes_train = read_labels_and_sizes(config.cascade_train)
    labels_val, sizes_val = read_labels_and_sizes(config.cascade_validation)
    labels_test, sizes_test = read_labels_and_sizes(config.cascade_test)

    # find the max length of sequences
    len_sequence = max(get_max_length(graphs_train),
                       get_max_length(graphs_val),
                       get_max_length(graphs_test))
    print("Max length of sequence:", len_sequence)

    print("Cascade graph embedding size:", config.gc_emd_size)
    print("Number of scale s:", config.number_of_s)

    print("Start writing train set into file.")
    write_cascade(graphs_train, labels_train, sizes_train, len_sequence,
                    config.train)
    print("Start writing validation set into file.")
    write_cascade(graphs_val, labels_val, sizes_val,
                    len_sequence,
                    config.val)
    print("Start writing test set into file.")
    write_cascade(graphs_test, labels_test, sizes_test, len_sequence,
                    config.test)

    time_end = time.time()
    print("Processing time: {0:.2f}s".format(time_end - time_start))
