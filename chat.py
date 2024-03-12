import chainlit as cl

from ctransformers import AutoModelForCausalLM


def get_prompt(instruction: str, history: list[str] | None = None) -> str:
    system = "You are an AI assitant that gives helpful concise anaswers."
    prompt = f"### System:\n{system}\n\n### User:\n"
    if len(history) > 0:
        prompt += f"This is the conversation history: {''.join(history)}. Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    return prompt


@cl.on_chat_start
def on_chat_start():
    cl.user_session.set("message_history", [])
    global llm
    llm = AutoModelForCausalLM.from_pretrained(
        "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
    )


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history")
    msg = cl.Message(content="")
    await msg.send()

    prompt = get_prompt(message.content, message_history)
    response = ""
    for word in llm(prompt, stream=True):
        await msg.stream_token(word)
        response += word
    await msg.update()
    message_history.append(response)


"""""
history = []
question = "the name of capital city of India is"

answer = ""
for word in llm(get_prompt(question), stream=True):
    print(word, end="", flush=True)
    answer += word
print()

history.append(answer)

question = "and which is the capital of the united states"

for word in llm(get_prompt(question, history), stream=True):
    print(word, end="", flush=True)
print()
"""
