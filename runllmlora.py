<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Chat with Qwen3-0.6B</title>
<style>
  body { font-family: Arial, sans-serif; margin: 20px; max-width: 700px; }
  textarea { width: 100%; height: 100px; font-family: monospace; font-size: 14px; }
  #response { white-space: pre-wrap; margin-top: 20px; background: #f0f0f0; padding: 10px; border-radius: 5px; }
  button { margin-top: 10px; padding: 8px 15px; font-size: 16px; }
</style>
</head>
<body>

<h2>Ask Qwen3-0.6B Model</h2>

<textarea id="inputText" placeholder="Type your instruction here..."></textarea><br />
<button id="sendBtn">Send</button>

<div id="response"></div>

<script>
async function sendChatCompletion(userInput) {
  const messages = [
    { role: "system", content: "You are a helpful assistant." },
    { role: "user", content: userInput }
  ];

  const response = await fetch("http://localhost:8000/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model: "/home/anuj/models/Qwen3-0.6B-Base/models--Qwen--Qwen3-0.6B-Base/snapshots/da87bfb608c14b7cf20ba1ce41287e8de496c0cd",
      messages: messages,
      max_tokens: 5000,
      temperature: 0.3,
      top_p: 0.9
    })
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HTTP error ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  return data.choices[0].message.content;
}

document.getElementById("sendBtn").addEventListener("click", async () => {
  const inputText = document.getElementById("inputText").value.trim();
  if (!inputText) {
    alert("Please enter an instruction.");
    return;
  }

  const responseDiv = document.getElementById("response");
  responseDiv.textContent = "Loading...";

  try {
    const answer = await sendChatCompletion(inputText);
    responseDiv.textContent = answer;
  } catch (error) {
    responseDiv.textContent = "Error: " + error.message;
  }
});
</script>

</body>
</html>
