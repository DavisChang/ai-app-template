# AI Chainlit App Template (RAG, QA)

Hi there, Developer! 👋 We're excited to have you on board. Chainlit is a powerful tool designed to help you prototype, debug and share applications built on top of LLMs.

## Useful Links 🔗

- **Documentation:** Get started with our comprehensive [Chainlit Documentation](https://docs.chainlit.io) 📚
- **Discord Community:** Join our friendly [Chainlit Discord](https://discord.gg/k73SQ3FyUh) to ask questions, share your projects, and connect with other developers! 💬

We can't wait to see what you create with Chainlit! Happy coding! 💻😊

## Welcome screen

To modify the welcome screen, edit the `chainlit.md` file at the root of your project. If you do not want a welcome screen, just leave this file empty.


## Run App

```bash
pip install -r requirements.txt

# Run to create vector database and add into database
python3 populate_database.py
python3 populate_database.py --reset

# Run chainlit app
chainlit run app.py

# code change
chainlit run app.py -w

# local ollama
chainlit run local_ollama.py -w
```