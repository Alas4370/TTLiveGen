import os
import torch
import gradio as gr
import asyncio
import threading
import nest_asyncio
from diffusers import DiffusionPipeline
from huggingface_hub import login
from dotenv import load_dotenv
from TikTokLive.client.client import TikTokLiveClient
from TikTokLive.events import ConnectEvent, CommentEvent, DisconnectEvent

# Apply Nest Asyncio for running TikTokLive in the same event loop
nest_asyncio.apply()

# Device Selection
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Hugging Face Token
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise ValueError("❌ Hugging Face token not found! Set it in the .env file.")

# Login to Hugging Face
login(hf_token)

# Initialize TikTok Live Client
client = TikTokLiveClient(unique_id="klaazikoo")
tiktok_connected = False

# Global Variables for UI Updates
image_path_to_display = None
prompt_status_to_display = "Waiting for a comment..."
current_pipeline = None

# Function to Update Gradio UI
def update_ui():
    return gr.update(value=image_path_to_display), gr.update(value=prompt_status_to_display)

# TikTok Live Events
@client.on(ConnectEvent)  
async def on_connect(event: ConnectEvent):
    global tiktok_connected
    tiktok_connected = True
    print("✅ Connected to live chat!")

@client.on(DisconnectEvent)
async def on_disconnect(event: DisconnectEvent):
    global tiktok_connected
    tiktok_connected = False
    print("❌ Disconnected. Attempting to reconnect...")
    asyncio.create_task(reconnect_tiktok()) 

async def reconnect_tiktok():
    """Tries to reconnect with an increasing delay."""
    global tiktok_connected
    delay = 3  

    while not tiktok_connected:
        print(f"⚠ Reconnecting in {delay} seconds...")
        await asyncio.sleep(delay)
        try:
            await client.run()
            return  
        except Exception as e:
            print(f"⚠ Reconnection failed: {e}")
            delay = min(delay * 2, 60) 

@client.on(CommentEvent)
async def on_comment(event: CommentEvent):
    global image_path_to_display, prompt_status_to_display

    if not event.comment.strip():
        print("⚠ Ignored empty comment")
        return

    print(f"{event.user.nickname}: {event.comment}")

    prompt_status_to_display = "Processing..."
    
    def process():
        global image_path_to_display, prompt_status_to_display
        image_path, status = generate_image(event.comment)
        image_path_to_display = image_path
        prompt_status_to_display = status

    await asyncio.to_thread(process)

    return gr.update(value=prompt_status_to_display), gr.update(value=image_path_to_display)

async def start_client():
    """Starts the TikTok Live client."""
    global tiktok_connected
    if not tiktok_connected:
        print("Attempting to connect...")
        try:
            await client.run()
        except Exception as e:
            print(f"⚠ Connection lost: {e}")
            await reconnect_tiktok()

def run_tiktok():
    """Runs the TikTok client in a separate thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(start_client())

def load_model():
    """Loads the Stable Diffusion model."""
    global current_pipeline
    try:
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        pipeline = DiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16, 
            variant="fp16", 
            token=hf_token
        )
        pipeline.to(device)
        pipeline.enable_attention_slicing()
        current_pipeline = pipeline
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")

# Image Generation

pipeline_lock = threading.Lock()
def generate_image(prompt, negative_prompt="blurry, dark, low quality", seed=0, cfg_scale=7, steps=30):
    """Generates an image while ensuring thread safety."""
    global current_pipeline

    if current_pipeline is None:
        return None, "⚠ Model not loaded"

    try:
        with pipeline_lock:  # Ensure only one thread accesses the model at a time
            generator = torch.manual_seed(seed)
            image = current_pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                guidance_scale=cfg_scale,
                num_inference_steps=steps,
                generator=generator
            ).images[0]

        image_path = os.path.abspath("generated_image.png")
        image.save(image_path)

        success_message = f"✅ Successfully generated: {prompt}"
        print(success_message)

        return image_path, success_message
    
    except Exception as e:
        error_message = f"❌ Error generating image: {str(e)}"
        print(error_message)
        return None, error_message

# Gradio UI
with gr.Blocks() as interface:
    gr.Markdown("# My own AI Playground -- Image Generator © 2025 4las__. ")

    prompt_status = gr.Textbox(label="Prompt Status", interactive=False)
    image_output = gr.Image(label="Generated Image")

    interface.load(fn=update_ui, inputs=[], outputs=[image_output, prompt_status]) 
    interface.queue() 

if __name__ == "__main__":
    load_model()  

    tiktok_thread = threading.Thread(target=run_tiktok, daemon=True)
    tiktok_thread.start()

    interface.launch()