import os
import torch
from transformers import SpeechT5ForTextToSpeech, SpeechT5Processor
import torchaudio
from torch.utils.data import Dataset, DataLoader

# Load the SpeechT5 model and processor
model_name = "microsoft/speecht5_tts"
processor = SpeechT5Processor.from_pretrained(model_name)
model = SpeechT5ForTextToSpeech.from_pretrained(model_name)

# Load dataset paths from a file
dataset_file = 'TTS_Fine_Tuning_Project/datasets/final_merged_dataset.txt'

# Function to read the dataset file and prepare the dataset
def read_dataset_file(file_path):
    audio_paths = []
    texts = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()  # Remove leading/trailing whitespace
            if line:  # Only process non-empty lines
                parts = line.split('|')
                if len(parts) == 2:  # Ensure we have exactly two parts
                    audio_path, text = parts
                    audio_path = audio_path.strip()  # Clean up whitespace
                    audio_paths.append(audio_path)
                    texts.append(text)
                else:
                    print(f"Skipping malformed line: {line}")  # Print the malformed line for debugging
    return audio_paths, texts

audio_paths, texts = read_dataset_file(dataset_file)

# Function to process data
def process_data(audio_paths, target_length):
    audio_data = []
    for audio_path in audio_paths:
        try:
            # Load the audio file
            audio, _ = torchaudio.load(audio_path)  # Load the audio file
            audio = audio.squeeze().numpy()  # Convert to numpy array
            
            # Pad or truncate audio to target length
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            if len(audio_tensor) < target_length:
                # Pad the audio if it's shorter
                audio_tensor = torch.nn.functional.pad(audio_tensor, (0, target_length - len(audio_tensor)))
            else:
                # Truncate the audio if it's longer
                audio_tensor = audio_tensor[:target_length]
            
            audio_data.append(audio_tensor.numpy())  # Convert back to numpy array
            print(f"Processed audio length: {len(audio_tensor)}")  # Print length for debugging
        except Exception as e:
            print(f"Error processing audio file {audio_path}: {e}")  # Catch any loading errors
    return audio_data

# Define target length for padding/truncating
target_length = 300  # Example length; adjust based on your data
audio_data = process_data(audio_paths, target_length)

# Create a custom dataset
class CustomDataset(Dataset):
    def __init__(self, audio_data, texts):
        self.audio_data = audio_data
        self.texts = texts

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        audio = self.audio_data[idx]
        text = self.texts[idx]
        inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True)
        audio_tensor = torch.tensor(audio, dtype=torch.float32)

        # Debugging statements
        print(f"Getting item {idx}: audio shape {audio_tensor.shape}, text: {text}")  # Debugging statement
        
        return {"input_ids": inputs["input_ids"].squeeze(), "audio": audio_tensor}

# Create DataLoader
dataset = CustomDataset(audio_data, texts)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Fine-tuning hyperparameters
learning_rate = 5e-5
num_epochs = 5
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Fine-tuning loop
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(model.device)
        audio = batch["audio"].to(model.device)

        # Forward pass
        try:
            outputs = model(input_ids=input_ids, audio=audio)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item()}")  # Print loss per batch
        except Exception as e:
            print(f"Error during training: {e}")

# Save the fine-tuned model
save_directory = "TTS_Fine_Tuning_Project/fine_tuned_speecht5"

try:
    # Ensure the parent directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)  # Create directory if it doesn't exist

    # Save the model and processor
    model.save_pretrained(save_directory)
    processor.save_pretrained(save_directory)

    print(f"Model and processor saved successfully in {save_directory}")

except Exception as e:
    print(f"Error during saving: {e}")

# Load the fine-tuned model and processor
try:
    processor = SpeechT5Processor.from_pretrained(save_directory)
    model = SpeechT5ForTextToSpeech.from_pretrained(save_directory)
    print("Model and processor loaded successfully.")
except Exception as e:
    print(f"Error loading the model and processor: {e}")
