import os

def find_file_directory(file_name, start_path="/home/"):
    for root, _, files in os.walk(start_path):
        for file in files:
            if file.startswith(file_name):
                return os.path.abspath(root)
    return None

# Find requried libraries and register them to PATH
def find_libraries():
    rbdl_name = "librbdl.so"
    urdfreader_name = "librbdl_urdfreader.so"

    rbdl_path = find_file_directory(rbdl_name)
    urdfreader_path = find_file_directory(urdfreader_name)

    if rbdl_path is not None:
        print(f"RBDL found at : {rbdl_path}")
        os.environ['LD_LIBRARY_PATH'] = f"{rbdl_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    if urdfreader_path is not None:
        print(f"URDF Reader found at : {urdfreader_path}")
        os.environ['LD_LIBRARY_PATH'] = f"{urdfreader_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

    print()
    return rbdl_path, urdfreader_path