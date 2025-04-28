# TTLiveGen

A real-time Generative AI system that connects to TikTok Live, where viewer comments become dynamic prompts for AI generation!
Built using Gradio, TensorFlow, PyTorch, CUDA, and other cutting-edge AI technologies.

🚀 Project Overview
This project captures live comments from a TikTok livestream and feeds them directly into a generative AI model.
The AI then creates visual or text-based outputs based on the user's prompt in real time, offering a new level of audience interaction and creativity.

Key Features:<br>
· 🧠 Generative AI using state-of-the-art models (TensorFlow/PyTorch)<br>
· 🎥 Live TikTok Integration via comment scraping<br>
· ⚡ CUDA Acceleration for real-time performance<br>
· 🌐 Gradio Interface for monitoring and showcasing outputs<br>
· 🛠️ Customizable model selection (image, text, or hybrid generation)<br>
· 🗨️ Instant AI reactions to viewer input<br>

🛠️ Technologies Used<br>
· Gradio — interactive UI<br>
· TensorFlow — machine learning framework<br>
· PyTorch — alternative ML framework for flexible modeling<br>
· CUDA — GPU acceleration<br>
· TikTok-Live-Connector — fetch TikTok Live comments (or custom scraping solution)<br>

⚙️ Important Reminder<br>
· You must create your own Hugging Face access token from huggingface.co.<br>
· After generating the token, add it to a .env file in your project root:<br>
    HUGGINGFACE_TOKEN=your_hf_token_here<br>
· Never share your .env file or token publicly.<br>
· Without a valid token, the application will not be able to download or access certain models.<br>

Optional (depending on model use):<br>
· Hugging Face Transformers<br>
· Stable Diffusion / Diffusers Library<br>
· Llama/Chatbot APIs (for text generation)<br>

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
