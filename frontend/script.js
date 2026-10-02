const questionInput = document.getElementById("question");
const askButton = document.getElementById("askButton");

const answerDiv = document.getElementById("answer");
const sourcesDiv = document.getElementById("sources");


askButton.addEventListener("click", askQuestion);


async function askQuestion() {

    const question = questionInput.value.trim();

    if (!question) {
        answerDiv.textContent = "Please enter a question.";
        return;
    }

    askButton.disabled = true;
    askButton.textContent = "Thinking...";

    answerDiv.innerHTML = '<p class="loading">Searching Tesla reports and generating an answer...</p>';

    sourcesDiv.innerHTML = "";


    try {

        const response = await fetch("http://127.0.0.1:8000/ask", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                question: question
            })

        });


        if (!response.ok) {
            throw new Error("Server error");
        }


        const data = await response.json();


        answerDiv.textContent = data.answer;


        data.sources.forEach((source) => {

            const sourceElement = document.createElement("div");

            sourceElement.className = "source";

            sourceElement.innerHTML = `
                <strong>Tesla ${source.year} 10-K</strong>
                <br>
                Page: ${source.page}
                <br>
                Chunk ID: ${source.chunk_id}
                <br>
                Similarity: ${source.similarity.toFixed(4)}
            `;

            sourcesDiv.appendChild(sourceElement);

        });

    }


    catch (error) {

        console.error(error);

        answerDiv.textContent =
            "Unable to connect to the Tesla RAG backend.";

    }


    finally {

        askButton.disabled = false;
        askButton.textContent = "Ask";

    }

}