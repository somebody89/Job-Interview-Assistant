import openai
import time
import gradio as gr
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

#models: Power = ft:gpt-4o-mini-2024-07-18:personal::Ad5i1jpL
#models: RF = ft:gpt-4o-mini-2024-07-18:personal::Ad64Rajc
#models: Control = ft:gpt-4o-mini-2024-07-18:personal::Ad6uZPTQ
#models: Hardware = ft:gpt-4o-mini-2024-07-18:personal::Ad6uucbS
#models: comm and networking = ft:gpt-4o-mini-2024-07-18:personal::Ad72m9wJ
#models: Software = ft:gpt-4o-mini-2024-07-18:personal::Ad72kLYQ
# Set the OpenAI API key
openai.api_key = "open ai key"
messages = []
messages_displayed = []
Start_time = 0
End_time = 0
Time_b = "Disable"
time1 = ""
# Function to generate a question
def reset_messages():
    global messages
    global messages_displayed
    messages = []
    messages_displayed = []
    messages.append({"role": "system", "content": "You are a helpful assistant who generates job interview questions."})


def get_model(topic):
    if topic == "Software Engineering":
        return "ft:gpt-4o-mini-2024-07-18:personal::Ad72kLYQ"

    if topic == "RF Engineering":
        return "ft:gpt-4o-mini-2024-07-18:personal::Ad64Rajc"

    if topic == "Communication and Networking Engineering":
        return "ft:gpt-4o-mini-2024-07-18:personal::Ad72m9wJ"

    if topic == "Hardware Engineering":
        return"ft:gpt-4o-mini-2024-07-18:personal::Ad6uucbS"

    if topic == "Control Engineering":
        return "ft:gpt-4o-mini-2024-07-18:personal::Ad6uZPTQ"

    if topic == "Power Engineering":
        return "ft:gpt-4o-mini-2024-07-18:personal::Ad5i1jpL"


def GPT(prompt,model="gpt-4o-mini"):
    global messages
    try:
        messages.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
            model=model,
            messages=messages
        )
        bot_response = response['choices'][0]['message']['content']
        messages.append({"role": "assistant", "content": bot_response})
        return bot_response
    except openai.error.OpenAIError as e:
        return f"An error occurred: {e}"



# Navigation functions

def response(message, chat_history):
    Prompt = """
You are a specialized job interview assistant. Your purpose is to help users prepare for job interviews. Here are the key behaviors you should exhibit given the user's message:"""+message+"""
m
1. **Greeting**: If the user greets you in any way (e.g., "hello," "hi," "hey"), respond with a friendly greeting and let them know you are a specialized job interview assistant.

   Example response:
   - "Hello! I'm your specialized job interview assistant. How can I help you prepare for your interview today?"

2. **Non-Job-Interview Questions**: If the user asks a question that is not related to job interviews (e.g., random questions, casual conversation), politely remind them to ask a job interview-related question.

   Example response:
   - "I’m here to help with job interview preparation! Please ask me a job interview-related question, and I'll be happy to assist you."

3. **Job Interview-Related Questions (General)**: If the user asks a job interview-related question that is not specific to the pre-defined engineering fields, generate an interview question or response relevant to the topic they mentioned.

   Example behavior:
   - User: "Can you ask me a medicine job interview question?"
     Bot: "Sure! Here’s a commonly asked interview question in medicine: 'Can you describe a challenging case you've managed and how you handled it?' Would you like help preparing an answer?"

4. **Specific Engineering Topics**: If the user asks about specific topics such as "Software Engineering," "Hardware Engineering," "Control Engineering," "Communication and Networking Engineering," "Power Engineering," or "RF Engineering," inform the user that there is another tab fine-tuned for these types of questions and suggest they explore it.

   Example response:
   - "I see you're interested in [Topic]. There's another tab fine-tuned for [Topic]-related questions. I suggest you test it out there for more specialized assistance!"

5. **Further Help**: After generating an example answer or question, ask the user if they would like further assistance, such as adapting the response to their personal experiences or receiving feedback on their answer.

   Example response:
   - "Does this answer help? If you'd like, I can help you adapt it to your experiences or provide tips for improvement."
"""

    bot_message = GPT(Prompt)
    chat_history.append((message, bot_message)) # Add bot message to chat history
    time.sleep(1)
    return "", chat_history


def go_to_page2(topic_input,difficulty_input,Time_input):
    if not topic_input.strip():
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "Please enter a valid topic!"
    try:
        prompt = "Provide a" + topic_input+ "job interview question categorized as" + difficulty_input + "difficult without\"\"."
        model = get_model(topic_input)
        question = GPT(prompt,model = model)
        global messages_displayed
        messages_displayed.append({"role": "user", "content": "Provide a question about "+topic_input+" with a difficulty of "+difficulty_input+"."})
        messages_displayed.append({"role": "assistant", "content": question.strip('\"')})
        # Remove any leading/trailing quotes if present and pass to the generated question box
        global Time_b
        Time_b = str(Time_input)
        global Start_time
        Start_time= time.time()
        question_cleaned = question.strip('\"')
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), question_cleaned
    except Exception as e:
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), f"Error: {e}"

# Function to analyze sentiment
def analyze_sentiment(user_answer):
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(user_answer)
    compound_score = sentiment_score['compound']
    if compound_score >= 0.05:
        return "Positive"
    elif compound_score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def go_to_page3(user_answer,Sentiment_input):
    messages_displayed.append({"role": "user", "content": user_answer})


    if Sentiment_input == "Enable":
        tone = analyze_sentiment(user_answer)
        ##prompt = (f"OK so I want you to take my answer: {user_answer} ""and first line say rating(/10)""then below that put in a title tips on improving your answer: ""So I want you to state teh feedback in bullet points without adding any other sentence but part of the feedback i want you to add a bullet point stating that I had a" + tone + "and if its not good how to fix it or if its good praise me just in 1 buller point as parrt of the feedback""After that, stop and ask me if I want to continue or proceed.(do not use bold)")
        prompt = f"""
    Take my answer: {user_answer}.
    On the first line, provide a rating (out of 10) based on the following criteria:
    - Clarity (3 points): How clearly and concisely my answer conveys the information.
    - Accuracy (3 points): How correct and relevant my answer is to the question.
    - Depth (2 points): Whether my answer demonstrates understanding and provides sufficient detail.
    - Engagement (2 points): How engaging or well-structured my answer is.

    Provide the score as a single number i dont want to see how it was calculted just show me the final number/10
    - Below that provide feedback in bullet points only and keep it clean only include the bullet points nothing else and after every point insert enter. Include one bullet point addressing my tone which is"""+ tone  +"""(e.g., if it was good, praise it; if it wasn’t, explain how to improve it).
    - Do not add any other sentences beyond the feedback.
    After providing the feedback, ask me if you want to ask me another question of difficulty based on my answer.
    """

    else:

      prompt = f"""
    Take my answer: {user_answer}.
    On the first line, provide a rating (out of 10) based on the following criteria:
    - Clarity (3 points): How clearly and concisely my answer conveys the information.
    - Accuracy (3 points): How correct and relevant my answer is to the question.
    - Depth (2 points): Whether my answer demonstrates understanding and provides sufficient detail.
    - Engagement (2 points): How engaging or well-structured my answer is.

    Provide the score as a single number i dont want to see how it was calculted just show me the final number/10
    - Below that provide feedback in bullet points only and keep it clean only include the bullet points nothing else and after every point insert enter.
    - Do not add any other sentences beyond the feedback.
    After providing the feedback, ask me if you want to ask me another question of difficulty based on my answer.
    """
    answer = GPT(prompt)
    messages_displayed.append({"role": "rating", "content": answer})
    split_answer = answer.split('\n', 1)
    if len(split_answer) < 2:
        score, feedback = answer, "No feedback provided."
    else:
        score, feedback = split_answer
    # Clear the user_answer textbox when moving to page 3 but leave the placeholder
    global time1
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), score, feedback,time1

def go_to_page2_from_page3(topic_input,difficulty_input):
    try:
        prompt = "Provide a" + topic_input+ "job interview question categorized as" + difficulty_input + "difficulty"+" based on the previous answer"
        model = get_model(topic_input)
        question = GPT(prompt, model = model)
        global Start_time
        Start_time = time.time()
        global messages_displayed
        messages_displayed.append({"role": "user", "content": "Provide a question about "+topic_input+" with a difficulty of "+difficulty_input+"."})
        messages_displayed.append({"role": "assistant", "content": question.strip('\"')})
        # Clear the user_answer textbox when going back to page 2
        return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False), question, gr.update(value="")
    except Exception as e:
        return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), f"Error: {e}", gr.update(value="")

def go_back_to_page1():
    reset_messages()
    return gr.update(visible=True), gr.update(visible=False), gr.update(visible=False), "", "", ""  # Reset textboxes
# Function to update the conversation history in the tab
def update_conversation_history():
    global messages_displayed
    # Join all messages in the conversation into a single string
    history = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages_displayed])
    return history
def end_session():
    # Hide all pages and show the thank you message
    return gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), "Thank you for using the Job Assistant!"
def update_time():
    global time1
    time1 = "timing is disabled"
    global Time_b
    if Time_b == "Enable":
      global Start_time
      global End_time
      End_time = time.time()
      elapsed_time = End_time - Start_time
      # Convert elapsed time to minutes and seconds
      minutes, seconds = divmod(elapsed_time, 60)

      # Round seconds to an integer
      minutes = int(minutes)
      seconds = int(seconds)


      time1 = "Elapsed time: "+str(minutes)+" minutes and "+str(seconds)+" seconds"

def generate(topic_input1):
    Prompt_ge = f"Generate exactly 5 job interview questions on the topic: {topic_input1}. Each question should be separated by a newline. Please do not include any additional information or context."
    model = get_model(topic_input1)
    question = GPT(Prompt_ge,model=model)
    question_list = [q.strip() for q in question.split("\n") if q.strip()]
    # Filter out empty lines in case the LLM generates any
    return question_list[:9]
def get_answer(selected_question,topic_input1):
    Prompt_ans = f"""
Answer the following interview question as if you were a candidate. Your answer should be concise and clear,
demonstrating a strong understanding of the topic. Aim to provide accurate and relevant information with sufficient detail,
while maintaining engagement and a well-structured response. Keep in mind the following criteria for scoring:

- **Clarity (3 points)**: How clearly and concisely your answer conveys the information.
- **Accuracy (3 points)**: How correct and relevant your answer is to the question.
- **Depth (2 points)**: Whether your answer demonstrates understanding and provides sufficient detail.
- **Engagement (2 points)**: How engaging or well-structured your answer is.

Make sure to only showcase your answer without any score and try giving me a 10/10 answer. Also make it in a paragrpah wihtout numbers or stars or points. Here’s the interview question:
{selected_question}
"""
    # Call your LLM API here (replace `GPT` with the actual API call function)
    model = get_model(topic_input1)
    answer = GPT(Prompt_ans, model = model)
    return answer


# Define the Gradio app
with gr.Blocks() as demo:
  with gr.Row():
    with gr.Tab("Job Interview Assistant:"):
    # Page 1 Layout
      with gr.Group(visible=True) as page1:
          gr.Markdown("# Job Interview Question Generator")
          gr.Markdown("Provide a topic, and the assistant will generate a related job interview question.")
          with gr.Row():
            topic_input = gr.Dropdown(choices=["Software Engineering", "Control Engineering", "Communication and Networking Engineering", "Hardware Engineering","RF Engineering", "Power Engineering"], label="Select a topic" ,scale = 2)
            difficulty_input = gr.Dropdown(choices=["Easy", "Medium", "Hard"], label="Select Difficulty:",scale = 2)
          with gr.Row():
            Sentiment_input = gr.Dropdown(choices=["Enable", "Disable"], label="Select Sentiment:",scale = 2)
            Time_input = gr.Dropdown(choices=["Enable", "Disable"], label="Select Time:",scale = 2)


          generate_button = gr.Button("Generate Question")
          end_button = gr.Button("End Session")  # Button to end the session

    # Page 2 Layout
      with gr.Group(visible=False) as page2:
        gr.Markdown("### Generated Question")
        generated_question = gr.Textbox(
            label="Generated Question",
            placeholder="The generated question will appear here.",
            interactive=False
        )
        user_answer = gr.Textbox(
            label="Your Answer",
            placeholder="Write your answer here..."
        )

        send_button = gr.Button("Send Answer")
        back_button_to_page1_from_page2 = gr.Button("Back to Page 1")

    # Page 3 Layout
      with gr.Group(visible=False) as page3:
        gr.Markdown("### Feedback and Score")
        score_box = gr.Textbox(
            label="Your Score",
            placeholder="The score will appear here.",
            interactive=False
        )
        feedback_box = gr.Textbox(
            label="Feedback",
            placeholder="The feedback will appear here.",
            interactive=False
        )
        time_box = gr.Textbox(
            label="Time",
            placeholder="The time taken will appear here.",
            interactive=False
        )
        next_question_button = gr.Button("Next Question")
        back_button_to_page1_from_page3 = gr.Button("Back to Page 1")
    # Create a new tab for displaying old messages
      with gr.Tab("Conversation History"):
        conversation_history = gr.Textbox(
        label="Conversation History",
        placeholder="This section displays your previous conversation...",
        interactive=False,
        show_label=True
    )
    # Thank You Page Layout
      with gr.Group(visible=False) as thank_you_page:
        gr.Markdown("### Thank you for using the Job Assistant!")
        gr.Markdown("We hope you found the interview question generator helpful.")

    # Navigation Logic
      generate_button.click( go_to_page2, inputs=[topic_input,difficulty_input,Time_input], outputs=[page1, page2, page3, generated_question])

      send_button.click(update_time).then(go_to_page3, inputs=[user_answer, Sentiment_input], outputs=[page1, page2, page3, score_box, feedback_box,time_box]).then(
        update_conversation_history, outputs=[conversation_history])

      next_question_button.click(
        go_to_page2_from_page3,
        inputs=[topic_input,difficulty_input],
        outputs=[page1, page2, page3, generated_question, user_answer]
    ).then(
    update_conversation_history, outputs=[conversation_history])  # Update history after clicking


      back_button_to_page1_from_page2.click(
        go_back_to_page1,
        inputs=[],
        outputs=[page1, page2, page3, generated_question, user_answer, score_box]
    ).then(
    update_conversation_history, outputs=[conversation_history])  # Update history after clicking

      back_button_to_page1_from_page3.click(
        go_back_to_page1,
        inputs=[],
        outputs=[page1, page2, page3, generated_question, user_answer, score_box]
    ).then(
    update_conversation_history, outputs=[conversation_history])  # Update history after clicking

      end_button.click(
        end_session,
        inputs=[],
        outputs=[page1, page2, page3, thank_you_page]
    )

    with gr.Tab("Question/Answer Bot:"):
            gr.Markdown("Select a topic, and the assistant will provide you with 9 questions:")

            topic_input1 = gr.Dropdown(choices=["Software Engineering", "Control Engineering", "Communication and Networking Engineering", "Hardware Engineering","RF Engineering", "Power Engineering"], label="Select a topic")

    # Button to generate questions based on the selected topic
            generate_btn = gr.Button("Generate Questions")

    # Placeholder for the questions (Radio button group)
            question_list = gr.Radio(choices=[], label="Select a question", interactive=True)

    # Placeholder for the answer
            answer_output = gr.Textbox(label="Answer", interactive=False)

    # Link the generate button to populate the question list
            def update_question_list(topic):
                questions = generate(topic)
                return gr.update(choices=questions)

            generate_btn.click(update_question_list, inputs=topic_input1, outputs=question_list)

    # Link the question list selection to display an answer
            question_list.change(get_answer, inputs=[question_list,topic_input1], outputs=answer_output)
    with gr.Tab("Chatbot"):
        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Type a message here...")
        clear = gr.ClearButton([msg, chatbot])
        msg.submit(response, [msg, chatbot], [msg, chatbot])


# Launch the Gradio app
demo.launch()
