import os

# Replace this with your building folder name if different
root_dir = "."

def print_tree(path, prefix=""):
    """Recursively print folders and files like a tree"""
    items = sorted(os.listdir(path))
    for i, item in enumerate(items):
        full_path = os.path.join(path, item)
        connector = "└── " if i == len(items) - 1 else "├── "
        print(prefix + connector + item)
        if os.path.isdir(full_path):
            extension = "    " if i == len(items) - 1 else "│   "
            print_tree(full_path, prefix + extension)

# Start printing
print(root_dir)
print_tree(root_dir)
