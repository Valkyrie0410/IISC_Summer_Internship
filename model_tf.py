import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class LinearEncoder(tf.keras.Model):
    def __init__(self, input_dim, bottleneck_dim):
        super(LinearEncoder, self).__init__()
        self.linear = layers.Dense(bottleneck_dim)

    def call(self, x):
        return self.linear(x)

class StochasticDecoder(tf.keras.Model):
    def __init__(self, bottleneck_dim, output_dim):
        super(StochasticDecoder, self).__init__()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.fc3 = layers.Dense(output_dim)
        self.log_var = tf.Variable(tf.zeros(output_dim))

    def call(self, x):
        x = tf.nn.relu(self.fc1(x))
        x = tf.nn.relu(self.fc2(x))
        mean = self.fc3(x)
        std = tf.math.exp(0.5 * self.log_var)
        eps = tf.random.normal(tf.shape(std))
        return mean + eps * std

class EncoderDecoder(tf.keras.Model):
    def __init__(self, input_dim, bottleneck_dim, output_dim):
        super(EncoderDecoder, self).__init__()
        self.encoder = LinearEncoder(input_dim, bottleneck_dim)
        self.decoder = StochasticDecoder(bottleneck_dim, output_dim)

    def call(self, x):
        bottleneck = self.encoder(x)
        reconstructed = self.decoder(bottleneck)
        return bottleneck, reconstructed

def read_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    frames = []
    i = 0
    while i < len(lines):
        num_atoms = 128
        i += 2  # Skip the atom count line and the comment line
        frame = []
        for _ in range(num_atoms):
            if i < len(lines):
                atom_line = lines[i].strip().split()
                # Fix the unpacking error by checking the length of atom_line
                if len(atom_line) >= 4:
                    x, y, z = map(float, atom_line[1:])
                    frame.append([x, y, z])
                    i += 1
        frames.append(frame)

    return np.array(frames)

file_path = '/home/smart/Downloads/animation_1.xyz'
frames = read_xyz(file_path)

frames_test = frames[1001:2000]
frames = frames[:1000]

flattened_coordinates = frames.reshape(frames.shape[0], -1)
flattened_coordinates_test = frames_test.reshape(frames_test.shape[0], -1)

def compute_pairwise_distances(frames):
    num_frames, num_atoms, _ = frames.shape
    pairwise_distances = []
    for frame in frames[:1000]:
        distances = []
        for i in range(128):
            for j in range(i, 128):
                if j != i:
                    dist = np.linalg.norm(frame[i] - frame[j])
                    distances.append(dist)
                    j += 3
        pairwise_distances.append(distances)
    return np.array(pairwise_distances)

pairwise_distances = compute_pairwise_distances(frames)
pairwise_distances_test = compute_pairwise_distances(frames_test)

features = np.hstack([flattened_coordinates, pairwise_distances])
mean = np.mean(features, axis=0)
std = np.std(features, axis=0)
normalized_features = (features - mean) / std

features_test = np.hstack([flattened_coordinates_test, pairwise_distances_test])
mean = np.mean(features_test, axis=0)
std = np.std(features_test, axis=0)
normalized_features_test = (features_test - mean) / std

input_data = tf.constant(normalized_features, dtype=tf.float32)
test_data = tf.constant(normalized_features_test, dtype=tf.float32)

input_dim = input_data.shape[1]  # Dimensionality of the input features
bottleneck_dim = 1  # Bottleneck dimension (can be tuned)
output_dim = input_dim  # Output dimension should match input dimension for reconstruction

model = EncoderDecoder(input_dim, bottleneck_dim, output_dim)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

num_epochs = 2000
for epoch in range(num_epochs):
    with tf.GradientTape() as tape:
        bottleneck, output = model(input_data)
        loss = tf.keras.losses.MeanSquaredError()(output, input_data)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss:.4f}')

test_bottleneck, test_output = model(test_data)
test_loss = tf.keras.losses.MeanSquaredError()(test_output, test_data)
print(test_output)
print(test_bottleneck)
print(f'Test Loss: {test_loss:.4f}')
