from fastapi import FastAPI, UploadFile, File, Form,Header, HTTPException
from typing import Optional
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

import aiofiles
import uvicorn
from io import BytesIO
from pathlib import Path
import json
import torch
import matplotlib
import requests
matplotlib.use('Agg')
import subprocess
import re
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize
from transformer_pose import TransformerSeq2Seq
# from tokenizers import Tokenizer
from vqtorch import VQVAE
from prediction import generate
import shutil
import os
import numpy as np
import tempfile
from natsort import natsorted
import cv2
import ffmpeg
import matplotlib.pyplot as plt
import logging
from datetime import datetime
from fastapi.staticfiles import StaticFiles
import stanza
import nemo.collections.asr as nemo_asr
import soundfile as sf
from deepmultilingualpunctuation import PunctuationModel
from contextlib import asynccontextmanager



# Logging setup (fix: set up logging before any logging calls, and set force=True)
log_dir = "Logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}-log.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True  # <-- ensures reconfiguration even if logging was set up elsewhere
)
logger = logging.getLogger(__name__)

app = FastAPI()

# === Model and NLP Initialization ===
# Use global variables and initialize in a startup event

asr_model = None
nlp = None
punct_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_model, nlp, punct_model
    # ASR Model
    MODEL_PATH = "/home/naren-root/Documents/SIP/Android-App/backend/tamil_stt.nemo"
    import nemo.collections.asr as nemo_asr
    asr_model = nemo_asr.models.EncDecCTCModel.restore_from(restore_path=MODEL_PATH)
    asr_model.eval().to("cuda")
    # NLP Models
    import stanza
    stanza.download('ta')
    nlp = stanza.Pipeline('ta', processors='tokenize,pos,mwt,lemma', use_gpu=False)
    from deepmultilingualpunctuation import PunctuationModel
    punct_model = PunctuationModel('ModelsLab/punctuate-indic-v1')
    yield
    # (No cleanup needed)

app = FastAPI(lifespan=lifespan)

video_path = Path("outputs/pose.mp4")
CHUNK_SIZE = 1024*1024
# Optionally, mount for static serving (uncomment if you want direct file access)
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")

def preprocess_audio(input_file: str) -> str:
    output_file = "/tmp/processed_audio.wav"
    cmd = ["ffmpeg", "-i", input_file, "-ac", "1", "-ar", "16000", "-sample_fmt", "s16", "-y", output_file]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_file

def transcribe_audio(audio_path: str, lang_id: str = "ta") -> str:
    global asr_model
    import soundfile as sf
    audio_data, _ = sf.read(audio_path)
    if len(audio_data.shape) > 1:
        return "Error: Audio is not mono"
    asr_model.cur_decoder = "rnnt"
    return asr_model.transcribe([audio_path], batch_size=1, logprobs=False, language_id=lang_id)[0]


# === Function: Text Preprocessing and Transformation ===
def punctuate_text(text):
    global punct_model
    restored_text = punct_model.restore_punctuation(text)
    restored_split_by_comma = restored_text.split(",")
    final_text = ""
    for sent in restored_split_by_comma:
        length_of_sentence = len(sent.split())
        if length_of_sentence <= 2:
            final_text += sent + ","
        elif sent[-1] != ".":
            final_text += sent + "."
        else:
            final_text += sent
    final_text = final_text.replace("!", "!.").replace("?", "?.").replace(",.", ",").replace("..",".").replace(" .", ".").replace(":", "")
    return final_text

def process(tamil_doc):
    processed_doc = []
    for sent in tamil_doc.sentences:
        processed_sentence = []
        for word in sent.tokens:
            is_mwt = len(word.id) != 1
            word_dict = word.to_dict()
            text = ""
            pos = ""
            all_items_in_feats = []
            if is_mwt:
                continue
            else:
                text = word_dict[0]["text"] or ""
                pos = word_dict[0]["upos"] or ""
                all_items_in_feats = word_dict[0].get("feats", "").split("|")
            all_items_in_feats_dict = dict()
            if all_items_in_feats and all_items_in_feats[0] != "":
                all_items_in_feats_dict = {x.split("=")[0] : x.split("=")[1] for x in all_items_in_feats}
            morph = all_items_in_feats_dict
            processed_sentence.append({"text" : text, "pos" : pos, "is_mwt" : is_mwt, "morph" : morph})
        processed_doc.append(processed_sentence)
    return processed_doc

def swap_words(doc, sentence_num, word_i, word_j):
    doc[sentence_num][word_i], doc[sentence_num][word_j] = doc[sentence_num][word_j], doc[sentence_num][word_i]
    return doc

def convert_svo_to_sov(doc): return doc

def convert_aux_before_verb(doc):
    for sent_num, sentence in enumerate(doc):
        for word_num, word in enumerate(sentence[:-1]):
            if word["pos"] == "AUX" and sentence[word_num+1]["pos"] == "VERB":
                doc = swap_words(doc, sent_num, word_num, word_num+1)
    return doc

def convert_adv_after_verb(doc):
    for sent_num, sentence in enumerate(doc):
        for word_num, word in enumerate(sentence[:-1]):
            if word["pos"] == "ADV" and sentence[word_num+1]["pos"] == "VERB":
                doc = swap_words(doc, sent_num, word_num, word_num+1)
    return doc

def convert_adj_after_noun(doc):
    for sent_num, sentence in enumerate(doc):
        for word_num, word in enumerate(sentence[:-1]):
            if word["pos"] == "ADJ" and sentence[word_num+1]["pos"] == "NOUN":
                doc = swap_words(doc, sent_num, word_num, word_num+1)
    return doc

def convert_num_after_noun(doc):
    for sent_num, sentence in enumerate(doc):
        for word_num, word in enumerate(sentence[:-1]):
            if word["pos"] == "NUM" and sentence[word_num+1]["pos"] == "NOUN":
                doc = swap_words(doc, sent_num, word_num, word_num+1)
    return doc

def is_negative(word):
    global nlp
    negative_root_list = "இல்லை இல்லாமல் இல்லாத இல்லாததால் இல்லாதிருந்தால் மாட்டேன் மாட்டாய் மாட்டான் மாட்டாள் மாட்டோம் மாட்டீர்கள் மாட்டார்கள் மாட்டல் வேண்டாம் வேண்டியதில்லை வேண்டாத வேண்டாதது முடியாது முடியவில்லை முடியாத முடியாமல் ஆகாது ஆகவில்லை ஆகாத ஆகாமல் அல்ல அல்லாத அல்லாமல் தெரியாது தெரியவில்லை தெரியாத தெரியாமல் பிடிக்காது பிடிக்கவில்லை பிடிக்காத பிடிக்காமல் போதாது போதவில்லை போதாத போதாமல் கூடாது கூடவில்லை கூடாத கூடாமல்".split(" ")
    sent_dict = nlp(word["text"])
    return any(i.text in negative_root_list for i in sent_dict.sentences[0].words)

def convert_neg_and_question_words(doc):
    punctuation_symbols = set("?.,!\"':")
    for sent_num, sentence in enumerate(doc):
        question_word, negative_word = None, None
        content_words, puncts = [], []
        for idx, word in enumerate(sentence):
            if word["text"] in punctuation_symbols:
                puncts.append((idx, word))
            else:
                content_words.append(word)

        for i, word in enumerate(content_words):
            if "PronType" in word.get("morph", {}) and word["morph"]["PronType"] == "Int":
                question_word = word

        for i, word in enumerate(content_words):
            if is_negative(word):
                negative_word = word

        filtered_words = [w for w in content_words if w not in [question_word, negative_word]]
        if negative_word: filtered_words.append(negative_word)
        if question_word: filtered_words.append(question_word)

        for idx, punct in puncts:
            if idx >= len(filtered_words): filtered_words.append(punct)
            else: filtered_words.insert(idx, punct)

        doc[sent_num] = filtered_words
    return doc

def extract_string_from_document(doc):
    final_string = ""
    for sentence in doc:
        for word in sentence:
            final_string += word["text"] + " "
    return final_string.strip()

# === Ordered grammar transformation rules ===
rule_list = [
    convert_svo_to_sov,
    convert_aux_before_verb,
    convert_adv_after_verb,
    convert_adj_after_noun,
    convert_num_after_noun,
    convert_neg_and_question_words
]

@app.get("/video")
async def video_endpoint(range: str = Header(None)):
    return FileResponse(video_path)
    

# Load models
device = "cuda" if torch.cuda.is_available() else "cpu"

vq = VQVAE().to(device)
vq.load_state_dict(torch.load("/home/naren-root/Documents/SIP/Custom-Impl/checkpoints1/vqvae_epoch_205.pt")['model_state_dict'])
vq.eval()

model = TransformerSeq2Seq(nhead=4, num_layers=4).to(device)
data = torch.load("/home/naren-root/Documents/SIP/Custom-Impl/checkpoints_transformer/transformer_model_overfit.pt")
model.load_state_dict(data["model_state"])
model.eval()

code_book = torch.load("codebook_512.pt", weights_only=False).to(device)


def visualise_pose(base_path, pose):
    # Ensure base_path is created
    if os.path.exists(base_path):
        shutil.rmtree(base_path)
    os.makedirs(base_path, exist_ok=True)
    pose = pose.detach().cpu().numpy()
    logging.info(f"POSE: {pose.shape} ")
    joint_index_map = {
        # Pose
        'NOSE': 0, 'LEFT_EYE_INNER': 1, 'LEFT_EYE': 2, 'LEFT_EYE_OUTER': 3,
        'RIGHT_EYE_INNER': 4, 'RIGHT_EYE': 5, 'RIGHT_EYE_OUTER': 6,
        'MOUTH_LEFT': 7, 'MOUTH_RIGHT': 8,
        'LEFT_SHOULDER': 9, 'RIGHT_SHOULDER': 10,
        'LEFT_ELBOW': 11, 'RIGHT_ELBOW': 12,
        'LEFT_WRIST': 13, 'RIGHT_WRIST': 14,
    
        # Left Hand
        'L_WRIST': 15, 'L_THUMB_CMC': 16, 'L_THUMB_MCP': 17, 'L_THUMB_IP': 18, 'L_THUMB_TIP': 19,
        'L_INDEX_FINGER_MCP': 20, 'L_INDEX_FINGER_PIP': 21, 'L_INDEX_FINGER_DIP': 22, 'L_INDEX_FINGER_TIP': 23,
        'L_MIDDLE_FINGER_MCP': 24, 'L_MIDDLE_FINGER_PIP': 25, 'L_MIDDLE_FINGER_DIP': 26, 'L_MIDDLE_FINGER_TIP': 27,
        'L_RING_FINGER_MCP': 28, 'L_RING_FINGER_PIP': 29, 'L_RING_FINGER_DIP': 30, 'L_RING_FINGER_TIP': 31,
        'L_PINKY_MCP': 32, 'L_PINKY_PIP': 33, 'L_PINKY_DIP': 34, 'L_PINKY_TIP': 35,
    
        # Right Hand
        'R_WRIST': 36, 'R_THUMB_CMC': 37, 'R_THUMB_MCP': 38, 'R_THUMB_IP': 39, 'R_THUMB_TIP': 40,
        'R_INDEX_FINGER_MCP': 41, 'R_INDEX_FINGER_PIP': 42, 'R_INDEX_FINGER_DIP': 43, 'R_INDEX_FINGER_TIP': 44,
        'R_MIDDLE_FINGER_MCP': 45, 'R_MIDDLE_FINGER_PIP': 46, 'R_MIDDLE_FINGER_DIP': 47, 'R_MIDDLE_FINGER_TIP': 48,
        'R_RING_FINGER_MCP': 49, 'R_RING_FINGER_PIP': 50, 'R_RING_FINGER_DIP': 51, 'R_RING_FINGER_TIP': 52,
        'R_PINKY_MCP': 53, 'R_PINKY_PIP': 54, 'R_PINKY_DIP': 55, 'R_PINKY_TIP': 56,
    }
    
    # Step 2: Pose connections
    pose_edges_orig = [
        ('NOSE','LEFT_EYE_INNER'), ('LEFT_EYE_INNER','LEFT_EYE'), ('LEFT_EYE','LEFT_EYE_OUTER'),
        ('NOSE','RIGHT_EYE_INNER'), ('RIGHT_EYE_INNER','RIGHT_EYE'), ('RIGHT_EYE','RIGHT_EYE_OUTER'),
        #('NOSE','MOUTH_LEFT'), ('NOSE','MOUTH_RIGHT'),
        ('LEFT_SHOULDER','RIGHT_SHOULDER'),
        ('LEFT_SHOULDER','LEFT_ELBOW'), ('LEFT_ELBOW','LEFT_WRIST'),
        ('RIGHT_SHOULDER','RIGHT_ELBOW'), ('RIGHT_ELBOW','RIGHT_WRIST')
    ]
    
    pose_connections = [(joint_index_map[a], joint_index_map[b]) for a, b in pose_edges_orig]
    
    # Step 3: Hand template for both hands
    hand_edges_template = [
        ('WRIST','THUMB_CMC'), ('THUMB_CMC','THUMB_MCP'), ('THUMB_MCP','THUMB_IP'), ('THUMB_IP','THUMB_TIP'),
        ('WRIST','INDEX_FINGER_MCP'), ('INDEX_FINGER_MCP','INDEX_FINGER_PIP'),
        ('INDEX_FINGER_PIP','INDEX_FINGER_DIP'), ('INDEX_FINGER_DIP','INDEX_FINGER_TIP'),
        ('WRIST','MIDDLE_FINGER_MCP'), ('MIDDLE_FINGER_MCP','MIDDLE_FINGER_PIP'),
        ('MIDDLE_FINGER_PIP','MIDDLE_FINGER_DIP'), ('MIDDLE_FINGER_DIP','MIDDLE_FINGER_TIP'),
        ('WRIST','RING_FINGER_MCP'), ('RING_FINGER_MCP','RING_FINGER_PIP'),
        ('RING_FINGER_PIP','RING_FINGER_DIP'), ('RING_FINGER_DIP','RING_FINGER_TIP'),
        ('WRIST','PINKY_MCP'), ('PINKY_MCP','PINKY_PIP'),
        ('PINKY_PIP','PINKY_DIP'), ('PINKY_DIP','PINKY_TIP'),
    ]
    
    left_hand_connections = [(joint_index_map[f'L_{a}'], joint_index_map[f'L_{b}']) for a, b in hand_edges_template]
    right_hand_connections = [(joint_index_map[f'R_{a}'], joint_index_map[f'R_{b}']) for a, b in hand_edges_template]
    
    # Step 4: Face regions
    face_regions = {
        "LEFT_EYE": [57, 58, 59, 60, 61, 62, 63, 64, 65, 66],
        "RIGHT_EYE": [67, 68, 69, 70, 71, 72, 73, 74, 75, 76],
        "MOUTH": [77, 78, 79, 80, 81, 82, 83, 84, 85, 86],
    }
    face_connections = []
    for region in face_regions.values():
        for i in range(len(region) - 1):
            face_connections.append((region[i], region[i + 1]))
    
    # Step 5: Drawing helper
    def draw_skeleton(ax, joints, connections, color='blue'):
        for start, end in connections:
            if start < len(joints) and end < len(joints):
                x = [joints[start][0], joints[end][0]]
                y = [-joints[start][1], -joints[end][1]] 
                ax.plot(x, y, c=color, linewidth=1)
    
    # Step 6: Render the pose
    def render_pose(base_path, frame, frame_num, save_path="pose_render"):
        
        fig, ax = plt.subplots(figsize=(6,6))
        save_path = os.path.join(base_path, save_path)
        x_vals = frame[:, 0]
        y_vals = -frame[:, 1]  
        # logging.info(f"X : {x_vals}")
        ax.scatter(x_vals, y_vals, c='red', s=5)
        
        draw_skeleton(ax, frame, pose_connections, color='blue')
        draw_skeleton(ax, frame, left_hand_connections, color='green')
        draw_skeleton(ax, frame, right_hand_connections, color='purple')
        draw_skeleton(ax, frame, face_connections, color='orange')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.title(f"Skeleton at frame number : {frame_num}")

        plt.savefig(f"{save_path}_{frame_num}.png", bbox_inches='tight', pad_inches=0)

        plt.close(fig)
    for i in range(pose.shape[0]):
        if i % 100 == 0:
            logging.info(f"Rendering pose {i}")
        render_pose(base_path, pose[i], i)
def pose_to_mp4(img_folder: str, output_path: str) -> str:
    images = [img for img in os.listdir(img_folder) if img.endswith('.png')]
    images = natsorted(images)

    if not images:
        raise ValueError("No frames to compile into video")

    first_path = os.path.join(img_folder, images[0])
    first = cv2.imread(first_path)
    if first is None:
        raise ValueError(f"Failed to read first image: {first_path}")
    h, w, _ = first.shape

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + "_raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(tmp_path, fourcc, 15, (w, h))
    if not out.isOpened():
        raise RuntimeError(f"VideoWriter failed to open. Ensure OpenCV supports 'mp4v' codec.")

    for img in images:
        frame_path = os.path.join(img_folder, img)
        frame = cv2.imread(frame_path)
        if frame is None:
            print(f"[WARN] Skipping unreadable image: {frame_path}")
            continue
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h))
        out.write(frame)

    out.release()

    # Check if OpenCV output exists
    if not os.path.exists(tmp_path) or os.path.getsize(tmp_path) == 0:
        raise RuntimeError("OpenCV video file creation failed")

    # Convert to proper MP4 with moov atom
    final_mp4 = output_path + ".mp4"
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", tmp_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        final_mp4
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg re-encoding failed: {e}")

    if not os.path.exists(final_mp4) or os.path.getsize(final_mp4) == 0:
        raise RuntimeError("Final video not created or is empty")

    return final_mp4
# Logging setup
log_dir = "Logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d')}-log.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

@app.post("/stt_audio")
async def stt_audio(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(await audio.read())
        temp_audio_path = temp_audio.name

    processed_audio = preprocess_audio(temp_audio_path)
    transcript = transcribe_audio(processed_audio)

    os.remove(temp_audio_path)
    os.remove(processed_audio)

    return {"transcription": transcript}


@app.post("/stt_video")
async def stt_video(video: UploadFile = File(...)):
    # Save uploaded video to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(await video.read())
        temp_video_path = temp_video.name

    # Extract audio as mono 16kHz wav using ffmpeg
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio_path = temp_audio.name

    ffmpeg_cmd = [
        "ffmpeg", "-i", temp_video_path,
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        "-y", temp_audio_path
    ]
    subprocess.run(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Use stt_audio logic to transcribe
    transcript = transcribe_audio(temp_audio_path)

    # Clean up
    os.remove(temp_video_path)
    os.remove(temp_audio_path)

    return {"transcription": transcript}

@app.post("/text_to_gloss")
async def text_to_gloss(text: str = Form(...)):
    global nlp
    punctuated_text = punctuate_text(text)
    doc = nlp(punctuated_text)
    processed_doc = process(doc)

    for rule in rule_list:
        processed_doc = rule(processed_doc)

    final_string = extract_string_from_document(processed_doc)
    return {"gloss": final_string}

@app.post("/gloss_to_pose")
async def gloss_to_pose(gloss: str = Form(...)):
    try:
        # Step 1: Tokenize gloss
        resp = requests.get("http://0.0.0.0:4444/encode", params={"text": gloss})
        resp.raise_for_status()
        encoded = resp.json()["tokens"]
        src = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to("cuda")
        # Step 2: Generate code sequence
        generated_code = generate(model, src.squeeze(0))  # shape: (seq_len,)
        tgt_seq = torch.tensor(generated_code, device=device)

        # Step 3: Get quantized vectors from codebook
        quantized = code_book[tgt_seq]  # shape: (seq_len, code_dim)
        quantized = quantized.unsqueeze(0).permute(0, 2, 1)  # shape: (1, code_dim, seq_len)

        # Step 4: Decode pose
        decoded_pose = vq.decoder(quantized).squeeze(0).detach().cpu().numpy()  # shape: (seq_len, pose_dim)

        return {"pose": decoded_pose.tolist()}

    except Exception as e:
        return {"error": str(e)}
    

@app.post("/gloss_to_video")
async def gloss_to_video(gloss: str = Form(...)):
    # ...convert gloss to video...
    return None

@app.post("/gloss_to_posevideo")
async def gloss_to_posevideo(gloss: str = Form(...)):
    try:
        logger.info("Received request for /gloss_to_posevideo")
        logger.info(f"Gloss input: {gloss}")

        # Tokenize and generate codes
        logger.info("Encoding gloss...")
        resp = requests.get("http://0.0.0.0:4444/encode", params={"text": gloss})
        resp.raise_for_status()
        encoded = resp.json()["tokens"]
        src = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to("cuda")
        logger.info(f"Encoded gloss: {src}")

        logger.info("Generating codes with transformer model...")
        gen_codes = generate(model, src.squeeze(0))
        tgt_seq = torch.tensor(gen_codes, device=device)
        logger.info(f"Generated codes: {gen_codes}")

        # Quantize and decode
        logger.info("Quantizing and decoding pose...")
        quantized = code_book[tgt_seq].unsqueeze(0).permute(0, 2, 1)
        decoded = vq.decoder(quantized).squeeze(0)
        logger.info(f"Decoded pose shape: {decoded.shape}")

        # Create temp workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            frames_dir = os.path.join(tmpdir, 'frames')
            logger.info(f"Rendering pose frames to {frames_dir}")
            visualise_pose(frames_dir, decoded)
            logger.info("Frames rendered, compiling video...")
            video_path = pose_to_mp4(frames_dir, os.path.join(tmpdir, 'output'))
            logger.info(f"Video created at {video_path}")

            # Save video permanently
            final_path = os.path.join("outputs", "pose.mp4")
            os.makedirs("outputs", exist_ok=True)
            shutil.copyfile(video_path, final_path)

            # Return video URL instead of file
            public_url = f"/outputs/pose.mp4"
            logger.info(f"Returning video URL: {public_url}")
            return JSONResponse(content={"url": public_url})

    except Exception as e:
        logger.error(f"Error in /gloss_to_posevideo: {e}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
@app.get("/retrieve_history")
async def retrieve_history():
    # ...get video and text...
    return None

@app.post("/store_data")
async def store_data(text: str = Form(...), video_path: str = Form(...)):
    # ...store video and text in history...
    return None

if __name__ == "__main__":
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",  # Listen on all interfaces for public access
        port=8000,
        reload=True
    )
