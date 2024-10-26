from flask import Flask, render_template, request
from langchain.llms import GPT4All
from langchain.tools import DuckDuckGoSearchTool
from langchain.agents import initialize_agent, Tool

# Инициализация GPT4All и DuckDuckGo Search
gpt4all = GPT4All(model_path="../models/gpt4all-j.bin")
search = DuckDuckGoSearchTool()

# Создаём цепочку инструментов для поиска
tools = [
    Tool(
        name="DuckDuckGo Search",
        func=search.run,
        description="Для поиска информации в интернете."
    )
]

# Инициализация агента LangChain
agent = initialize_agent(
    tools=tools,
    llm=gpt4all,
    agent="zero-shot-react-description",
    verbose=True
)

# Создание Flask приложения
app = Flask(__name__)

# Главная страница
@app.route("/", methods=["GET", "POST"])
def index():
    response = ""
    if request.method == "POST":
        query = request.form.get("query")
        response = agent.run(query)  # Запуск агента с введённым запросом
    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Запуск на всех интерфейсах сервера
