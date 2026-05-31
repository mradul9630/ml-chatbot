from chatbot.chatbot import get_response

print("=== NLP Chatbot ===")
print("Type 'exit' to quit")

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    response = get_response(user_input)
    print("Bot:", response)