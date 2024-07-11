import sqlite3
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import mdtraj as md
import math, itertools
from math import cos, sin, sqrt, acos, atan2, fabs, pi
# dcd_file = "/home/smart/Desktop/nico_data/Mg8/data/traj1/anim.dcd"
# dcd_file_1 = "/home/smart/Desktop/nico_data/Mg8/data/traj2/anim.dcd"
# dcd_file_2 = "/home/smart/Desktop/nico_data/Mg8/data/traj3/anim.dcd"
# pdb_file = "/home/smart/Desktop/nico_data/Mg8/4rum_Mg8_K150.pdb"
# traj = md.load(dcd_file, top=pdb_file)
# traj1 = md.load(dcd_file_1, top=pdb_file)
# traj2 = md.load(dcd_file_2, top=pdb_file)
# all_coordinates = traj.xyz[:, :279, :]
# all_coordinates = np.append(all_coordinates, traj1.xyz[:, :279, :], axis=0)
# all_coordinates = np.append(all_coordinates, traj2.xyz[:, :279, :], axis=0)
# print(all_coordinates.shape)
def load_full_dataset(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    cur.execute("SELECT * from data")
    coordinates = cur.fetchall()
    coordinates = np.array(coordinates).reshape(200001, 384)[1:]
    coordinates = coordinates.reshape(200000, 128, 3)
    # Load coordinates
    cur.execute("SELECT * FROM bond_lengths")
    bond_lengths= cur.fetchall()
    bond_lengths = np.array(bond_lengths).reshape(200001, 127)[1:]  

    cur.execute("SELECT * FROM bond_angles")
    bond_angles = cur.fetchall()
    bond_angles = np.array(bond_angles).reshape(200001, 188)[1:]

    cur.execute("SELECT * FROM dihedral_angles")
    dihedral_angles = cur.fetchall()
    dihedral_angles = np.array(dihedral_angles).reshape(200001, 125)[1:]
    
    # cur.execute("SELECT * FROM pairwise_distances")
    # pairwise = cur.fetchall()
    # pairwise = np.array(pairwise).reshape(200001, 1830)[1:]

    conn.close()
    bond_angles = np.radians(bond_angles)
    dihedral_angles = np.radians(dihedral_angles)
    features = np.hstack([bond_lengths, bond_angles, dihedral_angles])
    
    return features, coordinates


# def load_full_dataset(db_path):
#     conn = sqlite3.connect(db_path)
#     cur = conn.cursor()
    
#     # Load coordinates
#     cur.execute("SELECT * FROM data")
#     coordinates = cur.fetchall()
#     coordinates = np.array(coordinates).reshape(200001, 384)[1:]  
#     coordinates_1 = coordinates
#     selected_columns = []
#     for i in range(0, coordinates.shape[1], 6):
#         selected_columns.extend([i, i+1, i+2])

#     back_bone = coordinates[:, selected_columns]
#     back_bone = back_bone.reshape(200000, 64, 3)
#     coordinates_1 = coordinates_1.reshape(200000, 128, 3)
#     conn.close()
    
#     return back_bone, coordinates_1

def pairwise_distances(coordinates):
    pairwise_distances = []
    
    # Iterate over each set of coordinates
    for i in range(200000):
        distance = []
        for j in range(64):
            for k in range(j + 4, 64):
                dist = np.linalg.norm(coordinates[i][k] - coordinates[i][j])
                distance.append(dist)
        
        pairwise_distances.append(distance)
    
    return pairwise_distances

def radius_of_gyration(coordinates):
    coordinates_array = np.array(coordinates).reshape(200000, 64, 3)
    radius_of_gyrations = []
    for i in range(len(coordinates_array)):
        center_of_mass = np.mean(coordinates_array[i], axis=0)       
        distances_squared = np.sum((coordinates_array[i] - center_of_mass)**2, axis=1)        
        average_distance = np.mean(distances_squared)
        radius_of_gyration = np.sqrt(average_distance) 
        radius_of_gyrations.append(radius_of_gyration)

    radius_of_gyrations = np.array(radius_of_gyrations)
    radius_of_gyrations = radius_of_gyrations.reshape(100000, 1)
    
    return radius_of_gyrations

def calculate_bond_lengths(coordinates, bonds):
    bond_lengths = []
    for bond in bonds:
        atom1, atom2 = bond
        length = np.linalg.norm(coordinates[atom1] - coordinates[atom2])
        bond_lengths.append(length)
    return np.array(bond_lengths)

def calculate_bond_angles(coordinates, angles):
    bond_angles = []
    for angle in angles:
        atom1, atom2, atom3 = angle
        vec1 = coordinates[atom1] - coordinates[atom2]
        vec2 = coordinates[atom3] - coordinates[atom2]
        cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        bond_angles.append((angle_rad))
    return np.array(bond_angles)

def calculate_dihedral_angles(coordinates, dihedrals):
    dihedral_angles = []
    for dihedral in dihedrals:
        atom1, atom2, atom3, atom4 = dihedral
        b1 = coordinates[atom2] - coordinates[atom1]
        b2 = coordinates[atom3] - coordinates[atom2]
        b3 = coordinates[atom4] - coordinates[atom3]
        
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)
        
        m1 = np.cross(n1, b2/np.linalg.norm(b2))
        
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        
        dihedral_angle = np.arctan2(y, x)
        dihedral_angles.append((dihedral_angle))
    return np.array(dihedral_angles)
def find_BA(dd1, dd2, dd3, dd4):

    angleID = -1
    for aa in range(len(angles)):
        if (dd2 == angles[aa][0] and dd3 == angles[aa][1] and dd4 == angles[aa][2]):
            angleID = aa
            break
    if (angleID == -1):
        print("angle not found", dd2, dd3, dd4)
        exit()
        #find bond
    bondID = -1
    for bb in range(len(bonds)):
        if (dd3 == bonds[bb][0] and dd4 == bonds[bb][1]):
            bondID = bb
            break
    if (bondID == -1):
        print("bond not found")
        exit()

    return bondID, angleID

def place_atom(atom_a, atom_b, atom_c, angle, torsion, bond):

    R = bond
    ab = np.subtract(atom_b, atom_a)
    bc = np.subtract(atom_c, atom_b)
    bcn = bc / np.linalg.norm(bc)

    case = 1
    okinsert = False
    while (okinsert == False):
        #case 1
        if (case == 2):
            d = np.array([
                -R * cos(angle), R * cos(torsion) * sin(angle),
                R * sin(torsion) * sin(angle)
            ])
            n = np.cross(bcn, ab)
            n = n / np.linalg.norm(n)
            nbc = np.cross(bcn, n)
        #case 2
        elif (case == 1):
            d = np.array([
                -R * cos(angle), R * cos(torsion) * sin(angle),
                R * sin(torsion) * sin(angle)
            ])
            n = np.cross(ab, bcn)
            n = n / np.linalg.norm(n)
            nbc = np.cross(bcn, n)
        #case 3
        elif (case == 3):
            d = np.array([
                -R * cos(angle), R * cos(torsion) * sin(angle),
                -R * sin(torsion) * sin(angle)
            ])
            n = np.cross(ab, bcn)
            n = n / np.linalg.norm(n)
            nbc = np.cross(bcn, n)
        #case 4
        elif (case == 4):
            d = np.array([
                -R * cos(angle), R * cos(torsion) * sin(angle),
                R * sin(torsion) * sin(angle)
            ])
            n = np.cross(ab, bcn)
            n = n / np.linalg.norm(n)
            nbc = np.cross(n, bcn)

        m = np.array([[bcn[0], nbc[0], n[0]], [bcn[1], nbc[1], n[1]],
                        [bcn[2], nbc[2], n[2]]])
        d = m.dot(d)
        atom_d = d + atom_c

        #test dihedral
        r21 = np.subtract(atom_b, atom_a)
        r23 = np.subtract(atom_c, atom_b)
        r43 = np.subtract(atom_d, atom_c)
       
        n1 = np.cross(r21, r23)
        n2 = np.cross(r23, r43)
        
        n1 = n1 / np.linalg.norm(n1)
        n2 = n2 / np.linalg.norm(n2)
        r23 = r23 / np.linalg.norm(r23)
        m = np.cross(n1, r23)
        x = np.dot(n1, n2)
        y = np.dot(m, n2)
        phi = atan2(y, x)

        #test angle
        d12 = np.subtract(atom_b, atom_c)
        d32 = np.subtract(atom_d, atom_c)
        d12 = d12 / np.linalg.norm(d12)
        d32 = d32 / np.linalg.norm(d32)
        cos_theta = np.dot(d12, d32)
        m = np.linalg.norm(np.cross(d12, d32))
        theta = atan2(m, cos_theta)

        if (fabs(theta - angle) < 0.001 and fabs(phi - torsion) < 0.001):
            print("no case found", case, theta, angle, phi, torsion, atom_d)
            okinsert = True
        else:
            if (case < 4): case += 1
            else:
                break
    return atom_d

    ########################################################
def test_angle(atoms, angleID):
    ii, jj, kk = angles[angleID]
    d12 = np.subtract(atoms[ii], atoms[jj])
    d32 = np.subtract(atoms[kk], atoms[jj])
    d12 = d12 / np.linalg.norm(d12)
    d32 = d32 / np.linalg.norm(d32)
    cos_theta = np.dot(d12, d32)
    m = np.linalg.norm(np.cross(d12, d32))
    theta = np.arctan2(m, cos_theta)
    theta = np.degrees(theta)


    return theta

    ########################################################
def test_dihedral(atoms, dihedralID):

    ii, jj, kk, ll = dih[dihedralID]
    r21 = np.subtract(atoms[jj], atoms[ii])
    r23 = np.subtract(atoms[jj], atoms[kk])
    r43 = np.subtract(atoms[ll], atoms[kk])

    n1 = np.cross(r21, r23)
    n2 = np.cross(r23, r43)

    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)
    r23 = r23 / np.linalg.norm(r23)

    m = np.cross(n1, r23)
    x = np.dot(n1, n2)
    y = np.dot(m, n2)

    phi = atan2(y, x)

    return phi
    ########################################################
def new_config(CVsB, CVsA, CVsD):
    ang = CVsA[0]
    an = -1.0 * ang

    # Define rotation matrices
    R1 = np.array([[cos(an), -sin(an), 0.0],
                   [sin(an), cos(an), 0.0],
                   [0.0, 0.0, 1.0]])

    R2 = np.array([[1.0, 0.0, 0.0],
                   [0.0, cos(-math.pi / 4), -sin(-math.pi / 4)],
                   [0.0, sin(-math.pi / 4), cos(-math.pi / 4)]])

    R3 = np.array([[cos(-math.pi / 4), 0.0, sin(-math.pi / 4)],
                   [0.0, 1.0, 0.0],
                   [-sin(-math.pi / 4), 0.0, cos(-math.pi / 4)]])

    atoms = np.zeros((128, 3), float)

    # Initial vectors for atoms 1, 0, and 2
    vec10 = [1.0 / sqrt(2), 1.0 / sqrt(2), 0.0]
    vec20 = np.dot(R1, vec10)
    vec10 = np.dot(R2, vec10)
    vec20 = np.dot(R2, vec20)
    vec10 = np.dot(R3, vec10)
    vec20 = np.dot(R3, vec20)

    # Set positions for atoms 1, 0, and 2
    atoms[1] = [CVsB[0] * vec10[0], CVsB[0] * vec10[1], CVsB[0] * vec10[2]]
    atoms[0] = [0.0, 0.0, 0.0]  # Atom at index 0
    atoms[2] = [CVsB[1] * vec20[0], CVsB[1] * vec20[1], CVsB[1] * vec20[2]]

    # Iteratively place all other atoms based on dihedral angles
    for dd in range(len(dih)):
        dd1, dd2, dd3, dd4 = dih[dd]
        bondID, angleID = find_BA(dd1, dd2, dd3, dd4)
        
        coord = place_atom(atoms[dd1], atoms[dd2], atoms[dd3],
                           CVsA[angleID], CVsD[dd], CVsB[bondID])
        
        atoms[dd4] = coord
    
    return atoms

def remove_com(conf):
        # calculate center of mass165
    comp = [0.0, 0.0, 0.0]
    for i in range(len(conf)):
        for dim in range(3):
            comp[dim] += 1 * conf[i][dim]
    for dim in range(3):
        comp[dim] /= 3

        # substract center of mass
    conf_com = np.zeros((len(conf), 3), float)
    for i in range(len(conf)):
        for dim in range(3):
            conf_com[i, dim] = conf[i][dim] - comp[dim]

    return conf_com

def rotationmatrix(coordref, coord):

    assert (coordref.shape[1] == 3)
    assert (coordref.shape == coord.shape)
    correlation_matrix = np.dot(np.transpose(coordref), coord)
    vv, ss, ww = np.linalg.svd(correlation_matrix)
    is_reflection = (np.linalg.det(vv) * np.linalg.det(ww)) < 0.0
        #if is_reflection:
        #print "is_reflection"
        #vv[-1,:] = -vv[-1,:]
        #ss[-1] = -ss[-1]
        #vv[:, -1] = -vv[:, -1]
    rotation = np.dot(vv, ww)

    confnew = []
    for i in range(len(coord)):
        xx = rotation[0][0] * coord[i][0] + rotation[0][1] * coord[i][
            1] + rotation[0][2] * coord[i][2]
        yy = rotation[1][0] * coord[i][0] + rotation[1][1] * coord[i][
                1] + rotation[1][2] * coord[i][2]
        zz = rotation[2][0] * coord[i][0] + rotation[2][1] * coord[i][
                1] + rotation[2][2] * coord[i][2]
        confnew.append((xx, yy, zz))

    return confnew

db_path = "/home/smart/Documents/IISC/sqlite_1.db"
features, coordinates = load_full_dataset(db_path)
# num_atoms = 64
# num_frames = 200000
# pwd = pairwise_distances(back_bone)
# df = pd.DataFrame(pwd)
# df.to_csv('pairwise_distances.csv', index = False)
# rg = radius_of_gyration(coordinates)
# def compute_kde_pdf(bottleneck_values):
#     """Compute the PDF using kernel density estimation (KDE)."""
#     bottleneck_values = bottleneck_values.ravel()  # Ensure bottleneck_values is a 1D array
#     kde = gaussian_kde(bottleneck_values)
#     x_values = np.linspace(np.min(bottleneck_values), np.max(bottleneck_values), 1000)
#     pdf = kde(x_values)
#     return x_values, pdf

# def compute_free_energy(pdf, temperature=1):
#     k_B = 1  # Assume k_B = 1 for simplicity
#     free_energy = -k_B * temperature * np.log(pdf + 1e-10)  # Add a small constant to avoid log(0)
#     return free_energy

# x_grid, kde_values = compute_kde_pdf(rg)
# free_energy = compute_free_energy(kde_values)

# fig = plt.figure(figsize=(12, 8))
# plt.plot(x_grid, free_energy, linestyle = '-')
# plt.xlabel('Bottleneck Dimension 1')
# plt.ylabel('free energy')
# plt.title(f'Free Energy Distribution')
# plt.show()
num_frames = 200000
# bonds = [(i, i + 3) for i in range(2, 276, 3)]
# angles = [(i, i + 1, i + 2) for i in range(0, 276, 3)]
dih = [(1, 0, 2, 3), (1, 0, 2, 4), (3, 2, 4, 5), (3, 2, 4, 6), (5, 4, 6, 7), (5, 4, 6, 8),
 (7, 6, 8, 9), (7, 6, 8, 10), (9, 8, 10, 11), (9, 8, 10, 12), (11, 10, 12, 13), (11, 10, 12, 14),
 (13, 12, 14, 15), (13, 12, 14, 16), (15, 14, 16, 17), (15, 14, 16, 18), (17, 16, 18, 19), (17, 16, 18, 20),
 (19, 18, 20, 21), (19, 18, 20, 22), (21, 20, 22, 23), (21, 20, 22, 24), (23, 22, 24, 25), (23, 22, 24, 26),
 (25, 24, 26, 27), (25, 24, 26, 28), (27, 26, 28, 29), (27, 26, 28, 30), (29, 28, 30, 31), (29, 28, 30, 32),
 (31, 30, 32, 33), (31, 30, 32, 34), (33, 32, 34, 35), (33, 32, 34, 36), (35, 34, 36, 37), (35, 34, 36, 38),
 (37, 36, 38, 39), (37, 36, 38, 40), (39, 38, 40, 41), (39, 38, 40, 42), (41, 40, 42, 43), (41, 40, 42, 44),
 (43, 42, 44, 45), (43, 42, 44, 46), (45, 44, 46, 47), (45, 44, 46, 48), (47, 46, 48, 49), (47, 46, 48, 50),
 (49, 48, 50, 51), (49, 48, 50, 52), (51, 50, 52, 53), (51, 50, 52, 54), (53, 52, 54, 55), (53, 52, 54, 56),
 (55, 54, 56, 57), (55, 54, 56, 58), (57, 56, 58, 59), (57, 56, 58, 60), (59, 58, 60, 61), (59, 58, 60, 62),
 (61, 60, 62, 63), (61, 60, 62, 64), (63, 62, 64, 65), (63, 62, 64, 66), (65, 64, 66, 67), (65, 64, 66, 68),
 (67, 66, 68, 69), (67, 66, 68, 70), (69, 68, 70, 71), (69, 68, 70, 72), (71, 70, 72, 73), (71, 70, 72, 74),
 (73, 72, 74, 75), (73, 72, 74, 76), (75, 74, 76, 77), (75, 74, 76, 78), (77, 76, 78, 79), (77, 76, 78, 80),
 (79, 78, 80, 81), (79, 78, 80, 82), (81, 80, 82, 83), (81, 80, 82, 84), (83, 82, 84, 85), (83, 82, 84, 86),
 (85, 84, 86, 87), (85, 84, 86, 88), (87, 86, 88, 89), (87, 86, 88, 90), (89, 88, 90, 91), (89, 88, 90, 92),
 (91, 90, 92, 93), (91, 90, 92, 94), (93, 92, 94, 95), (93, 92, 94, 96), (95, 94, 96, 97), (95, 94, 96, 98),
 (97, 96, 98, 99), (97, 96, 98, 100), (99, 98, 100, 101), (99, 98, 100, 102), (101, 100, 102, 103), (101, 100, 102, 104),
 (103, 102, 104, 105), (103, 102, 104, 106), (105, 104, 106, 107), (105, 104, 106, 108), (107, 106, 108, 109), (107, 106, 108, 110),
 (109, 108, 110, 111), (109, 108, 110, 112), (111, 110, 112, 113), (111, 110, 112, 114), (113, 112, 114, 115), (113, 112, 114, 116),
 (115, 114, 116, 117), (115, 114, 116, 118), (117, 116, 118, 119), (117, 116, 118, 120), (119, 118, 120, 121), (119, 118, 120, 122),
 (121, 120, 122, 123), (121, 120, 122, 124), (123, 122, 124, 125), (123, 122, 124, 126), (125,124,126,127)]
bonds = [(1,0), (0,2), (2,3), (2, 4), (4, 5), (4, 6), (6, 7), (6, 8), (8, 9), (8, 10), (10, 11), (10, 12), (12, 13), (12, 14), (14, 15), (14, 16), (16, 17), (16, 18), (18, 19), (18, 20), (20, 21), (20, 22), (22, 23), (22, 24), (24, 25), (24, 26), (26, 27), (26, 28), (28, 29), (28, 30), (30, 31), (30, 32), (32, 33), (32, 34), (34, 35), (34, 36), (36, 37), (36, 38), (38, 39), (38, 40), (40, 41), (40, 42), (42, 43), (42, 44), (44, 45), (44, 46), (46, 47), (46, 48), (48, 49), (48, 50), (50, 51), (50, 52), (52, 53), (52, 54), (54, 55), (54, 56), (56, 57), (56, 58), (58, 59), (58, 60), (60, 61), (60, 62), (62, 63), (62, 64), (64, 65), (64, 66), (66, 67), (66, 68), (68, 69), (68, 70), (70, 71), (70, 72), (72, 73), (72, 74), (74, 75), (74, 76), (76, 77), (76, 78), (78, 79), (78, 80), (80, 81), (80, 82), (82, 83), (82, 84), (84, 85), (84, 86), (86, 87), (86, 88), (88, 89), (88, 90), (90, 91), (90, 92), (92, 93), (92, 94), (94, 95), (94, 96), (96, 97), (96, 98), (98, 99), (98, 100), (100, 101), (100, 102), (102, 103), (102, 104), (104, 105), (104, 106), (106, 107), (106, 108), (108, 109), (108, 110), (110, 111), (110, 112), (112, 113), (112, 114), (114, 115), (114, 116), (116, 117), (116, 118), (118, 119), (118, 120), (120, 121), (120, 122), (122, 123), (122, 124), (124, 125), (124, 126), (126, 127)]
angles = [(1, 0, 2), (0, 2, 3), (0, 2, 4), (3, 2, 4), (2, 4, 5), (2, 4, 6), (5, 4, 6), (4, 6, 7), (4, 6, 8),
 (7, 6, 8), (6, 8, 9), (6, 8, 10), (9, 8, 10), (8, 10, 11), (8, 10, 12), (11, 10, 12), (10, 12, 13), (10, 12, 14),
 (13, 12, 14), (12, 14, 15), (12, 14, 16), (15, 14, 16), (14, 16, 17), (14, 16, 18), (17, 16, 18), (16, 18, 19), (16, 18, 20),
 (19, 18, 20), (18, 20, 21), (18, 20, 22), (21, 20, 22), (20, 22, 23), (20, 22, 24), (23, 22, 24), (22, 24, 25), (22, 24, 26),
 (25, 24, 26), (24, 26, 27), (24, 26, 28), (27, 26, 28), (26, 28, 29), (26, 28, 30), (29, 28, 30), (28, 30, 31), (28, 30, 32),
 (31, 30, 32), (30, 32, 33), (30, 32, 34), (33, 32, 34), (32, 34, 35), (32, 34, 36), (35, 34, 36), (34, 36, 37), (34, 36, 38),
 (37, 36, 38), (36, 38, 39), (36, 38, 40), (39, 38, 40), (38, 40, 41), (38, 40, 42), (41, 40, 42), (40, 42, 43), (40, 42, 44),
 (43, 42, 44), (42, 44, 45), (42, 44, 46), (45, 44, 46), (44, 46, 47), (44, 46, 48), (47, 46, 48), (46, 48, 49), (46, 48, 50),
 (49, 48, 50), (48, 50, 51), (48, 50, 52), (51, 50, 52), (50, 52, 53), (50, 52, 54), (53, 52, 54), (52, 54, 55), (52, 54, 56),
 (55, 54, 56), (54, 56, 57), (54, 56, 58), (57, 56, 58), (56, 58, 59), (56, 58, 60), (59, 58, 60), (58, 60, 61), (58, 60, 62),
 (61, 60, 62), (60, 62, 63), (60, 62, 64), (63, 62, 64), (62, 64, 65), (62, 64, 66), (65, 64, 66), (64, 66, 67), (64, 66, 68),
 (67, 66, 68), (66, 68, 69), (66, 68, 70), (69, 68, 70), (68, 70, 71), (68, 70, 72), (71, 70, 72), (70, 72, 73), (70, 72, 74),
 (73, 72, 74), (72, 74, 75), (72, 74, 76), (75, 74, 76), (74, 76, 77), (74, 76, 78), (77, 76, 78), (76, 78, 79), (76, 78, 80),
 (79, 78, 80), (78, 80, 81), (78, 80, 82), (81, 80, 82), (80, 82, 83), (80, 82, 84), (83, 82, 84), (82, 84, 85), (82, 84, 86),
 (85, 84, 86), (84, 86, 87), (84, 86, 88), (87, 86, 88), (86, 88, 89), (86, 88, 90), (89, 88, 90), (88, 90, 91), (88, 90, 92),
 (91, 90, 92), (90, 92, 93), (90, 92, 94), (93, 92, 94), (92, 94, 95), (92, 94, 96), (95, 94, 96), (94, 96, 97), (94, 96, 98),
 (97, 96, 98), (96, 98, 99), (96, 98, 100), (99, 98, 100), (98, 100, 101), (98, 100, 102), (101, 100, 102), (100, 102, 103), (100, 102, 104),
 (103, 102, 104), (102, 104, 105), (102, 104, 106), (105, 104, 106), (104, 106, 107), (104, 106, 108), (107, 106, 108), (106, 108, 109), (106, 108, 110),
 (109, 108, 110), (108, 110, 111), (108, 110, 112), (111, 110, 112), (110, 112, 113), (110, 112, 114), (113, 112, 114), (112, 114, 115), (112, 114, 116),
 (115, 114, 116), (114, 116, 117), (114, 116, 118), (117, 116, 118), (116, 118, 119), (116, 118, 120), (119, 118, 120), (118, 120, 121), (118, 120, 122),
 (121, 120, 122), (120, 122, 123), (120, 122, 124), (123, 122, 124), (122, 124, 125), (122, 124, 126), (125, 124, 126), (124, 126, 127)]



atom = new_config(features[0][:127], features[0][127:315], features[0][315:])
coord = remove_com(coordinates[0])
ba = calculate_bond_angles(atom, angles)
bl = calculate_bond_lengths(atom, bonds)
di = calculate_dihedral_angles(atom, dih)
print(bl," ",features[0][:127])
print(ba," ",features[0][127:315])
print(di," ",features[0][315:])


# bond_angles = np.zeros((num_frames, len(angles)))
# bond_lengths = np.zeros((num_frames, len(bonds)))
# dihedral_angles = np.zeros((num_frames, len(dihedrals)))
# print(len(angles))
# print(len(bonds))
# print(len(dihedrals))
# for i in range(num_frames):
#     frame_coordinates = coordinates[i]
#     bond_length = calculate_bond_lengths(frame_coordinates, bonds)
#     bond_angle = calculate_bond_angles(frame_coordinates, angles)
#     dihedral_angle = calculate_dihedral_angles(frame_coordinates, dihedrals)
#     bond_lengths[i] = bond_length
#     bond_angles[i] = bond_angle 
#     dihedral_angles[i] = dihedral_angle 

# df = pd.DataFrame(bond_lengths)
# df.to_csv('bond_lengths.csv', index = False)

# df = pd.DataFrame(bond_angles)
# df.to_csv('bond_angles.csv', index = False)

# df = pd.DataFrame(dihedral_angles)
# df.to_csv('dihedral_angles.csv', index = False)