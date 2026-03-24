To Run this project just follow step by step process.
1. Clone the repo
2. Create your own .env  (.env) and Open .env and replace the real key:
GEMINI_API_KEY=your_gemini_api_key_here  https://aistudio.google.com/
3. Run docker build and docker run

   follow this command:
   1.git clone https://github.com/yourusername/fresh2.git
2. cd fresh2
3. cp .env.example .env
# edit .env with their own Gemini API key
4. docker build -t fresh2-app .
5. docker run -p 8000:8000 --env-file .env fresh2-app

   NOTE: I add all documentation along this project
