import os
from pydub import AudioSegment
from pydub.effects import normalize
from gtts import gTTS  # Google Text-to-Speech

# Pronunciation dictionary for technical terms
pronunciation_dict = {
    "Version control": "vur-zhuhn kuhn-trohl",
    "Git": "git",
    "Cybersecurity": "sai-bur-si-kyoor-i-tee",
    "Data integrity": "day-tuh in-teg-ri-tee",
    "Blockchain": "blok-chain",
    "Data mining": "day-tuh mai-ning",
    "Digital transformation": "dij-i-tuhl trans-faw-may-shun",
    "APIs": "ay-pee-eyes",  
    "Microservices": "mai-kroh-ser-vuh-siz",
    "embedded systems": "em-bed-id sis-tems",
    "Natural Language Processing": "nach-er-uhl lang-gwihj proh-ses-ing",
    "Cloud computing": "kloud kuhm-pyu-ting",
    "Network protocols": "net-wurk proh-tuh-kawls",
    "python": "pai-thon",
    "data visualisation": "day-tuh vizh-oo-uh-luh-zay-shun",
    "artificial intelligence": "ahr-tih-fish-uhl in-tel-i-juhns",
    "quantum computing": "kwon-tuhm kuhm-pyu-ting",
    "HTML": "eych-tee-em-el",
    "REST": "rest",
    "Java": "jah-vah",
    "JavaScript": "jah-vah-script",
    "Data structures": "day-tuh struhk-churz",
    "Deep learning": "deep lur-ning",
    "software": "sawft-wair",
    "machine learning": "muh-sheen lur-ning",
    "CUDA": "koo-duh",
    "OAuth": "oh-auth",
}

# Function to get the phonetic representation of a term
def get_phonetic_representation(term):  
    return pronunciation_dict.get(term, term)

# Define the input directories for audio files
input_directory_1 = "TTS_Fine_Tuning_Project/datasets/audio_files"
input_directory_2 = "TTS_Fine_Tuning_Project/datasets/LJ_Speech/LJSpeech-1.1/wavs"
output_directory = "TTS_Fine_Tuning_Project/output_audio_files"
dataset_file = "TTS_Fine_Tuning_Project/datasets/final_merged_dataset.txt"

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Load sentences from the dataset file
def load_sentences(dataset_file):
    sentences = {}
    with open(dataset_file, 'r') as file:
        for line in file:
            parts = line.strip().split('|')
            if len(parts) == 2:
                file_path = os.path.basename(parts[0].strip())
                sentence = parts[1].strip()

                # Replace technical terms with their phonetic representations
                for term in pronunciation_dict.keys():
                    if term in sentence:
                        phonetic_term = get_phonetic_representation(term) 
                        sentence = sentence.replace(term, phonetic_term) 

                sentences[file_path] = sentence
    return sentences

# Generate audio using TTS if audio file not found
def generate_audio_from_sentence(sentence, output_file):
    try:
        tts = gTTS(sentence)
        temp_mp3 = output_file.replace('.wav', '.mp3')
        tts.save(temp_mp3)

        # Convert MP3 to WAV format using pydub
        audio = AudioSegment.from_mp3(temp_mp3)
        audio.export(output_file, format='wav')

        os.remove(temp_mp3)

        print(f"Generated TTS audio: {output_file}")
        return output_file  
    except Exception as e:
        print(f"Error generating audio for sentence: '{sentence}'. {e}")
        return None

# Fine-tuning function for audio
def fine_tune_audio(audio):  
    audio = audio + 15  
    audio = normalize(audio)  
    audio = audio.low_pass_filter(3000)  
    audio = audio.high_pass_filter(200)   
    return audio

# Load sentences from the dataset file
sentences = load_sentences(dataset_file)

# List to store missing audio files for TTS generation
missing_files = []
sequence_number = 1  

# Loop through each audio file that has a corresponding sentence
for audio_file in sentences.keys():  
    audio_file_path_1 = os.path.join(input_directory_1, audio_file)
    audio_file_path_2 = os.path.join(input_directory_2, audio_file)

    if os.path.isfile(audio_file_path_1):
        audio_file_path = audio_file_path_1
    elif os.path.isfile(audio_file_path_2):
        audio_file_path = audio_file_path_2
    else:
        print(f"Audio file {audio_file} not found. Generating fine-tuned audio using TTS...")
        generated_audio_name = f"audio_seq{sequence_number}.wav"
        generated_audio_path = os.path.join(output_directory, generated_audio_name)
        audio_file_path = generate_audio_from_sentence(sentences[audio_file], generated_audio_path)

        if not audio_file_path:
            missing_files.append(audio_file)
            continue

        sequence_number += 1  

    try:
        audio = AudioSegment.from_file(audio_file_path)
        tuned_audio = fine_tune_audio(audio)
        output_path = os.path.join(output_directory, audio_file)
        tuned_audio.export(output_path, format='wav')

        print(f"Successfully processed and saved: {audio_file} with sentence: '{sentences[audio_file]}'")
    
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")

# Log the missing files for which TTS was used
if missing_files:
    print("\nThe following audio files were not found and TTS was used instead:")
    for file in missing_files:
        print(f"- {file}")
else:
    print("\nAll audio files were successfully found and processed.")
