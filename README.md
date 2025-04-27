# TTLiveGen

A real-time Generative AI system that connects to TikTok Live, where viewer comments become dynamic prompts for AI generation!
Built using Gradio, TensorFlow, PyTorch, CUDA, and other cutting-edge AI technologies.

🚀 Project Overview
This project captures live comments from a TikTok livestream and feeds them directly into a generative AI model.
The AI then creates visual or text-based outputs based on the user's prompt in real time, offering a new level of audience interaction and creativity.

Key Features:
· 🧠 Generative AI using state-of-the-art models (TensorFlow/PyTorch)
· 🎥 Live TikTok Integration via comment scraping
· ⚡ CUDA Acceleration for real-time performance
· 🌐 Gradio Interface for monitoring and showcasing outputs
· 🛠️ Customizable model selection (image, text, or hybrid generation)
· 🗨️ Instant AI reactions to viewer input

🛠️ Technologies Used
· Gradio — interactive UI
· TensorFlow — machine learning framework
· PyTorch — alternative ML framework for flexible modeling
· CUDA — GPU acceleration
· TikTok-Live-Connector — fetch TikTok Live comments (or custom scraping solution)

⚙️ Important Reminder
· You must create your own Hugging Face access token from huggingface.co.
· After generating the token, add it to a .env file in your project root:
    HUGGINGFACE_TOKEN=your_hf_token_here
· Never share your .env file or token publicly.
· Without a valid token, the application will not be able to download or access certain models.

Optional (depending on model use):
· Hugging Face Transformers
· Stable Diffusion / Diffusers Library
· Llama/Chatbot APIs (for text generation)

🔥 How It Works
1. The script connects to your TikTok Live session.
2. Viewer comments are fetched in real-time.
3. Each comment is used as a prompt for the AI model.
4. The AI generates an output.
5. Outputs are displayed via Gradio or streamed back to TikTok.

📜 License
This project is licensed under the MIT License.
Feel free to use, modify, and contribute!

🙌 Acknowledgments
Special thanks to the open-source community behind TensorFlow, PyTorch, Gradio, TikTok Live API projects, and Hugging Face.
