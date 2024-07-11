import numpy as np 
import pandas as pd 
import sqlite3
import torch
import mdtraj as md

dcd_file = "/home/smart/Desktop/nico_data/Mg8/data/traj1/anim.dcd"
pdb_file = "/home/smart/Desktop/nico_data/Mg8/4rum_Mg8_K150.pdb"
traj = md.load(dcd_file, top=pdb_file)
all_coordinates = traj.xyz[:, :279, :]
all_coordinates = all_coordinates.reshape(-1, 837)

def read_xyz(file_path, file_path_1):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    frames = []
    i = 0
    while i < len(lines):
        num_atoms = 128
        i += 2  # Skip the atom count line and the comment line
        frame = []
        for _ in range(num_atoms):
            if i<len(lines):
                atom_line = lines[i].strip().split()
            # Fix the unpacking error by checking the length of atom_line
            if len(atom_line) >= 4:
                    x, y, z = map(float, atom_line[1:])
            frame.append([x, y, z])
            i += 1
        frames.append(frame)

    with open(file_path_1, 'r') as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        num_atoms = 128
        i += 2  # Skip the atom count line and the comment line
        frame = []
        for _ in range(num_atoms):
            if i<len(lines):
                atom_line = lines[i].strip().split()
            # Fix the unpacking error by checking the length of atom_line
            if len(atom_line) >= 4:
                    x, y, z = map(float, atom_line[1:])
            frame.append([x, y, z])
            i += 1
        frames.append(frame)

    return np.array(frames)

def compute_and_store_pairwise_distances(frames, db_path):
    num_frames, num_atoms, _ = frames.shape

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM pairwise_distances")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames_tensor = torch.tensor(frames, device=device, dtype=torch.float32)

    frame_id = 0
    for frame in frames_tensor:
        distance_id = 0
        for i in range(num_atoms):
            for j in range(i, num_atoms):  # Iterate with a step of 4 as per original logic
                if i == j:
                    j = j+4
                if j < num_atoms:
                    dist = torch.norm(frame[i] - frame[j]).item()
                    cursor.execute("INSERT INTO pairwise_distances (frame_id, distance_id, distance) VALUES (?, ?, ?)",
                               (frame_id, distance_id, dist))
                    distance_id += 1
        frame_id += 1

    conn.commit()
    conn.close()

def findvec(frames):
    num_frames, num_atoms, _ = frames.shape
    vectors = []
    for frame in range(num_frames):
        vec = []
        for i in range(1, num_atoms):
            dist = np.linalg.norm(frames[frame, i] - frames[frame, i - 1])
            vec.append(dist)
        vectors.append(vec)
    return np.array(vectors)

# file_path = '/home/smart/Downloads/animation_1.xyz'
# file_path_1 = '/home/smart/Downloads/animation_2.xyz'
# db_path = '/home/smart/Documents/IISC/sqlite_1.db'
# frames = read_xyz(file_path, file_path_1)

# # vectors = findvec(frames)
# # vectors_1 = vectors.reshape(vectors.shape[0], -1)
# # df = pd.DataFrame(vectors_1)
# # df.to_csv('vectors.csv', index = False)
# # pairwise_distances = compute_and_store_pairwise_distances(frames, db_path)

# frames_1 = frames.reshape(frames.shape[0], -1)
# df = pd.DataFrame(frames_1)
# df.to_csv('data.csv', index = False)

