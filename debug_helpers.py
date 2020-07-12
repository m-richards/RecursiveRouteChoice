


def print_sparse(mat, round_=6):
    for x in range(mat.shape[0]):
        for y in range(mat.shape[1]):
            if mat[x, y] != 0:
                print(f"{(x + 1, y + 1)}: {mat[x, y]:.6g}")
    try:

        print("nnz", mat.count_nonzero())
    except Exception:
        pass


def print_data_struct(network_data_struct):
    for i, j in zip(network_data_struct.data_fields, network_data_struct.data_array):
        print(i, "=========\n")
        print_sparse(j)

    print("incidence mat with nonzero arcs")
    print(network_data_struct.incidence_matrix)
