import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout


# Load and preprocess the dataset
def preprocess_gait_data(file_path, sequence_length=30):
    import pandas as pd

    # Load the dataset
    gait_data = pd.read_csv(file_path)

    # Combine disorder columns into a single target variable
    gait_data['Disorder'] = gait_data[
        ['Disorder_ALS', 'Disorder_Normal', 'Disorder_hunt', 'Disorder_park']
    ].idxmax(axis=1)

    # Map disorder columns to class labels
    disorder_mapping = {
        "Disorder_ALS": "ALS",
        "Disorder_Normal": "Normal",
        "Disorder_hunt": "Huntington",
        "Disorder_park": "Parkinson"
    }
    gait_data['Disorder'] = gait_data['Disorder'].map(disorder_mapping)

    # Encode target labels
    label_encoder = LabelEncoder()
    gait_data['Disorder_Label'] = label_encoder.fit_transform(gait_data['Disorder'])

    # Select numerical features for training
    numerical_features = [
        "Left Stride Interval (sec)", "Right Stride Interval (sec)",
        "Left Swing Interval (sec)", "Right Swing Interval (sec)",
        "Left Swing Interval (% of stride)", "Right Swing Interval (% of stride)",
        "Left Stance Interval (sec)", "Right Stance Interval (sec)",
        "Double Support Interval (sec)", "Age", "Height(m)", "Weight(kg)",
        "GaitSpeed(m/sec)"
    ]

    # Normalize features
    scaler = MinMaxScaler()
    gait_data[numerical_features] = scaler.fit_transform(gait_data[numerical_features])

    # Group data by Subject_ID and convert to sequences
    features = gait_data[numerical_features].values
    labels = gait_data['Disorder_Label'].values
    subject_ids = gait_data['Subject_ID']

    def create_sequences(features, labels, subject_ids, seq_length):
        sequences, sequence_labels = [], []
        unique_subjects = subject_ids.unique()
        for subject in unique_subjects:
            subject_data = features[subject_ids == subject]
            subject_labels = labels[subject_ids == subject]
            for i in range(len(subject_data) - seq_length + 1):
                sequences.append(subject_data[i:i + seq_length])
                sequence_labels.append(subject_labels[i + seq_length - 1])
        return np.array(sequences), np.array(sequence_labels)

    X, y = create_sequences(features, labels, gait_data['Subject_ID'], sequence_length)
    return X, y, label_encoder


# Load dataset
file_path = 'C:\\Users\\Orbit\\Downloads\\gait_dataset.csv'
sequence_length = 30
X, y, label_encoder = preprocess_gait_data(file_path, sequence_length)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Build the Simple RNN model
num_classes = len(np.unique(y))
input_shape = X.shape[1:]

model = Sequential([
    SimpleRNN(64, activation='tanh', input_shape=input_shape),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')  # Output layer for classification
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Save the model
model.save('gait_rnn_model.h5')

# Save model architecture and weights as CSV
# Save the model architecture
with open('gait_rnn_model_architecture.json', 'w') as json_file:
    json_file.write(model.to_json())

# Save weights as CSV
weights = model.get_weights()
import pandas as pd

for i, weight in enumerate(weights):
    pd.DataFrame(weight).to_csv(f'weight_{i}.csv', index=False)

# Decode labels for interpretation
class_labels = label_encoder.inverse_transform(np.arange(num_classes))
print("Class Labels:", class_labels)

# Save evaluation results
with open('evaluation_metrics.csv', 'w') as f:
    f.write(f"Test Loss,Test Accuracy\n{test_loss},{test_accuracy}")
