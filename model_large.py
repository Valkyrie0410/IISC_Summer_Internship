import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mdtraj as md
import sqlite3
from torch.utils.data import Dataset, DataLoader
import torch.nn.init as init
import matplotlib.pyplot as plt
# def create_table_with_every_10th_record(db_path, original_table, new_table):
#     conn = sqlite3.connect(db_path)
#     cursor = conn.cursor()

#     # Create the new table
#     cursor.execute(f"""
#     CREATE TABLE IF NOT EXISTS {new_table} (
#         frame_id INTEGER,
#         distance_id INTEGER,
#         distance REAL
#     );""")

#     cursor.execute("DELETE FROM pairwise_distances_new")

#     for i in range(0, 100000, 10):  # Assuming your frame_ids range from 0 to 100000
#         cursor.execute(f"""
#         INSERT INTO {new_table} (frame_id, distance_id, distance)
#         SELECT frame_id, distance_id, distance
#         FROM {original_table}
#         WHERE frame_id = ?;
#         """, (i,))

#     conn.commit()
#     conn.close()



# class PolymerDataset(Dataset):
#     def __init__(self, db_path, batch_size):
#         self.db_path = db_path
#         self.batch_size = batch_size

#     def __len__(self):
#         return self.batch_size

#     def __getitem__(self, idx):
#         conn = sqlite3.connect("sqlite_1.db")
#         cur = conn.cursor()
#         sql = f"""
#         SELECT frame_id FROM pairwise_distances_new ORDER BY random() LIMIT {batch_size}"""
#         sql_1 = f"""SELECT distance FROM pairwise_distances_new WHERE frame_id = ?"""
#         sql_2 = f"""SELECT * FROM data"""
#         cur.execute(sql_2)
#         coordinates = cur.fetchall()
#         coordinates = np.array(coordinates)
#         coordinates = coordinates.reshape(100001, 384)
#         pairwise_distances = []
#         flattened_coordinates =[]
#         cur.execute(sql)
#         frame_ids = cur.fetchall()
#         i =0;
#         for frame_id in frame_ids:
#             cur.execute(sql_1, frame_id)
#             flattened_coordinates.append(coordinates[frame_id])
#             alldis = cur.fetchall()
#             pairwise_distances.append(alldis)
#             i+=1
#         flattened_coordinates = np.array(flattened_coordinates)
#         pairwise_distances = np.array(pairwise_distances)
#         pairwise_distances = pairwise_distances.reshape(i, 8252)
#         features = np.hstack([flattened_coordinates, pairwise_distances])
#         mean = np.mean(features, axis=0)
#         std = np.std(features, axis=0)
#         normalized_features = (features - mean) / std  
#         input_data = torch.tensor(normalized_features, dtype=torch.float32)
#         conn.close()
#         return input_data


def compute_mean_std(db_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Load coordinates
    sql = """SELECT * FROM data"""
    cur.execute(sql)
    coordinates = cur.fetchall()
    coordinates = np.array(coordinates)
    coordinates = coordinates.reshape(100001, 384)[1:] # Adjust shape if needed
    coordinates = coordinates[::10]
    # Load pairwise distances
    sql = """SELECT distance FROM pairwise_distances_new"""
    cur.execute(sql)
    pairwise_distances = cur.fetchall()
    pairwise_distances = np.array(pairwise_distances).reshape(10000, 8252)  # Adjust shape if needed

    # Combine features
    features = np.hstack([coordinates, pairwise_distances])

    # Move features to GPU
    features_tensor = torch.tensor(features, dtype=torch.float32).cuda()

    # Compute mean and std on GPU
    mean = torch.mean(features_tensor, dim=0)
    std = torch.std(features_tensor, dim=0)

    conn.close()
    
    return mean.cpu().numpy(), std.cpu().numpy()

mean, std = compute_mean_std("sqlite_1.db")

class PolymerDataset(Dataset):
    def __init__(self, db_path, batch_size, preload_data=True, device='cpu', mean=mean, std=std):
        self.db_path = db_path
        self.batch_size = batch_size
        self.device = device
        self.preload_data = preload_data
        self.mean = mean 
        self.std = std 

        if self.preload_data:
            self.coordinates, self.pairwise_distances = self._preload_data()
        
    def __len__(self):
        return self.batch_size

    def _preload_data(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        # Load coordinates
        cur.execute("SELECT * FROM data")
        coordinates = cur.fetchall()
        coordinates = np.array(coordinates).reshape(100001, 384)[1:]

        # Select every 10th row
        coordinates = coordinates[::10]

        # Load pairwise distances
        cur.execute("SELECT frame_id, distance_id, distance FROM pairwise_distances_new")
        rows = cur.fetchall()
        conn.close()

        pairwise_distances = np.zeros((10001, 8252))  
        for frame_id, distance_id, distance in rows:
            if frame_id % 10 == 0:
                pairwise_distances[frame_id // 10, distance_id] = distance

        return coordinates, pairwise_distances

    def __getitem__(self, idx):
        if self.preload_data:
            frame_indices = np.random.choice(self.coordinates.shape[0], self.batch_size, replace=False)
            flattened_coordinates = self.coordinates[frame_indices]
            pairwise_distances = self.pairwise_distances[frame_indices]
        else:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()

            # Fetch random frame_ids
            cur.execute(f"SELECT frame_id FROM pairwise_distances_new ORDER BY random() LIMIT {self.batch_size}")
            frame_ids = [row[0] for row in cur.fetchall()]

            # Fetch coordinates and distances
            flattened_coordinates = []
            pairwise_distances = []

            cur.execute("SELECT * FROM data")
            coordinates = cur.fetchall()
            coordinates = np.array(coordinates).reshape(100001, 384)[1:]
            coordinates = coordinates[::10]

            for frame_id in frame_ids:
                flattened_coordinates.append(coordinates[frame_id])
                cur.execute(f"SELECT distance_id, distance FROM pairwise_distances_new WHERE frame_id = {frame_id}")
                distances = cur.fetchall()
                distance_array = np.zeros(8252)
                for distance_id, distance in distances:
                    distance_array[distance_id] = distance
                pairwise_distances.append(distance_array)

            flattened_coordinates = np.array(flattened_coordinates)
            pairwise_distances = np.array(pairwise_distances)

            conn.close()

        # Combine and normalize features
        features = np.hstack([flattened_coordinates, pairwise_distances])
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0) # Avoid division by zero by setting zero std to one
        normalized_features = (features - mean) / (std)  # Add a small epsilon to avoid zero division
        
        # Convert to tensor and move to the specified device
        input_data = torch.tensor(normalized_features, dtype=torch.float32).to(self.device)
        
        return input_data



class LinearEncoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim):
        super(LinearEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, bottleneck_dim)

    def forward(self, x):
        return self.linear(x)

class StochasticDecoder(nn.Module):
    def __init__(self, bottleneck_dim, output_dim, variance = 0.005):
        super(StochasticDecoder, self).__init__()
        self.fc1 = nn.Linear(bottleneck_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        # self.fc4 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, output_dim)
        self.log_var = nn.Parameter(torch.zeros(output_dim))
        self.elu = nn.ELU()
        self.variance = variance

        init.uniform_(self.fc1.weight, -0.005, 0.005)
        init.uniform_(self.fc2.weight, -0.005, 0.005)
        # init.uniform_(self.fc4.weight, -0.005, 0.005)
        init.uniform_(self.fc3.weight, -0.005, 0.005)

    def forward(self, x):
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        # x = self.elu(self.fc4(x))
        mean = self.fc3(x)
        # std = torch.exp(0.5 * self.log_var)
        # eps = torch.randn_like(mean)
        # noise = eps * std * self.variance# Gaussian noise
        return mean 

class EncoderDecoder(nn.Module):
    def __init__(self, input_dim, bottleneck_dim, output_dim, noise_variance=0.005):
        super(EncoderDecoder, self).__init__()
        self.encoder = LinearEncoder(input_dim, bottleneck_dim)
        self.decoder = StochasticDecoder(bottleneck_dim, output_dim)
        self.noise_variance = noise_variance

    def forward(self, x):
        bottleneck = self.encoder(x)
        noise = torch.randn_like(bottleneck) * self.noise_variance
        bottleneck_noisy = bottleneck + noise
        reconstructed = self.decoder(bottleneck_noisy)
        return bottleneck, reconstructed

db_path = "/home/smart/Documents/IISC/sqlite_1.db"
# original_table = 'pairwise_distances'
# new_table = 'pairwise_distances_new'
# create_table_with_every_10th_record(db_path, original_table, new_table)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
train_dataset = PolymerDataset(db_path, batch_size, preload_data=True, device=device)
val_dataset = PolymerDataset(db_path, batch_size, preload_data=True, device=device)  # Use a separate validation set if available

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

sample = train_dataset[0]
input_dim = sample.shape[1]  # Dimensionality of the input features
print(input_dim)
bottleneck_dim = 1  # Bottleneck dimension (can be tuned)
output_dim = input_dim  # Output dimension should match input dimension for reconstruction

model = EncoderDecoder(input_dim, bottleneck_dim, output_dim)

criterion = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.003)

model.to(device)

train_losses = []
val_losses = []

num_epochs = 11
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for data in train_dataloader:
        input_data = data
        input_data = input_data.to(device)
        bottleneck, output = model(input_data)
        loss = criterion(output, input_data)

    # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_dataloader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in val_dataloader:
            input_data = data.to(device)
            bottleneck, output = model(input_data)
            loss = criterion(output, input_data)
            val_loss += loss.item()
    
    val_loss /= len(val_dataloader)
    val_losses.append(val_loss)



    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()