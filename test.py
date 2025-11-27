import h5py

h5_path = "./dataset/ukdale/ukdale.h5"

with h5py.File(h5_path, 'r') as f:
    def print_name(name):
        print(name)
    f.visit(print_name)
