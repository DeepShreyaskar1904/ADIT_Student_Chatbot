# import json
# import nltk
# import random
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.pipeline import make_pipeline
# import openai
# import os
#
# nltk.download('punkt')
#
#
#
# # Sample training data
# data = {
#     "admission": ["How to apply?", "What are the admission requirements?", "When is the deadline?"],
#     "courses": ["What courses are available?", "Tell me about the syllabus", "Which courses are offered?"],
#     "fees": ["What is the tuition fee?", "How much does the program cost?", "What are the payment options?"],
#     "exam": ["When is the next exam?", "How to register for exams?", "Exam schedule"],
#     "HOD-AI department":["who is the Head of AI department ?","who is the HOD of AI department"],
#     "HOD-IT department":["who is the Head of IT department?","who is the HOD of IT department"],
#     "who":["who are you?"],
#     "Hi":["hi","hello","hey"],
#     "time":["what is the time of college"],
#     "who-dinesh":["who is Dinesh prajapati","who is DJP","Who is dinesh"]
# }
#
# responses = {
#     "Hi":["hello, i am your chatbot. how may i help you !!"],
#     "admission": "You can apply online through our portal. The deadline is June 30.",
#     "courses": "We offer a variety of courses including CS, Business, and Arts.",
#     "fees": "The tuition fee depends on the course. Please visit our finance section.",
#     "exam": "The next exam schedule is published on the website.",
#     "HOD-AI department":"Dinesh Prajapati",
#     "HOD-IT department":"Narendra chauhan",
#     "who":"I am an amazing chatbot of ADIT college developed by Dhruval Chauhan And Deep Shreyaskar.",
#     "time":"college starts at 9:00 AM to 5:00 PM",
#     "who-dinesh":"Dr.dinesh J Prajapti is Head of AI department"
#
#
# }
#
# # Preprocess data
# X, y = [], []
# for intent, phrases in data.items():
#     for phrase in phrases:
#         X.append(phrase)
#         y.append(intent)
#
# # Train model
# model = make_pipeline(CountVectorizer(), MultinomialNB())
# model.fit(X, y)
#
# # Save the model
# import joblib
# joblib.dump(model, "chatbot_model.pkl")
#
# # Function to predict response
# def get_response(user_input):
#     model = joblib.load("chatbot_model.pkl")
#     intent = model.predict([user_input])[0]
#     return responses.get(intent, "Sorry, I don't understand. Please contact support.")
#
#
#
#
# # Set up OpenAI API key
import json
import nltk
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import openai
import os
import joblib

# Download necessary resources
nltk.download('punkt')

# Sample training data
data = {
    "admission": [
        "How to apply?",
        "What are the admission requirements?",
        "When is the deadline?",
        "What is the admission process?",
        "Can I apply after the deadline?",
        "How can I check the status of my application?",
        "Are there any entrance exams for admission?"
    ],
    "courses": [
        "What courses are available?",
        "Tell me about the syllabus",
        "Which courses are offered?",
        "What courses should I take for computer science?",
        "Is there any course for Data Science?",
        "What is the duration of the programs?",
        "Are there any online courses offered?"
    ],
    "fees": [
        "What is the tuition fee?",
        "How much does the program cost?",
        "What are the payment options?",
        "Are there any scholarships available?",
        "Can I pay the fees in installments?",
        "Are there any additional fees?",
        "What are the refund policies for tuition?"
    ],
    "exam": [
        "When is the next exam?",
        "How to register for exams?",
        "Exam schedule",
        "Where can I find my exam timetable?",
        "What is the format of the exams?",
        "Can I reschedule my exam?",
        "Are exams online or offline?"
    ],
    "HOD-AI department": [
        "Who is the Head of AI department?",
        "Who is the HOD of AI department?",
        "Can you tell me about the AI department head?",
        "Who is in charge of the AI department?",
        "Is there a professor for AI courses?",
        "Who supervises the AI research?"
    ],
    "HOD-IT department": [
        "Who is the Head of IT department?",
        "Who is the HOD of IT department?",
        "Can you tell me about the IT department head?",
        "Who is in charge of the IT department?",
        "Who supervises the IT courses?"
    ],
    "who": [
        "Who are you?",
        "What is your purpose?",
        "Who created you?",
        "Tell me more about yourself?",
        "What can you do?",
        "Are you a student or faculty?"
    ],
    "Hi": [
        "Hi",
        "hii",
        "hiii",
        "Hello",
        "Hey",
        "Greetings",
        "Good Morning",
        "How are you?"
    ],
    "time": [
        "What is the time of college?",
        "When does the college start?",
        "What time does the college open?",
        "What is the daily schedule for college?",
        "What are the working hours of the college?",
        "When is the lunch break?"
    ],
    "location": [
        "Where is the college located?",
        "Where can I find ADIT College?",
        "What is the address of ADIT College?",
        "How do I reach ADIT College?",
        "Is ADIT College located in the city?"
    ],
    "facilities": [
        "What facilities are available on campus?",
        "Is there a library in the college?",
        "Does the college have sports facilities?",
        "Are there any hostels available?",
        "Is there a cafeteria?",
        "Does the college have Wi-Fi?"
    ],
    "internships": [
        "Are there any internship opportunities?",
        "How can I apply for internships?",
        "Are internships paid?",
        "Does the college help in getting internships?",
        "When should I start applying for internships?",
        "Do students get placement assistance?"
    ],
    "placements": [
        "Does the college offer placement assistance?",
        "What companies visit for campus placements?",
        "What is the placement record of the college?",
        "How can I prepare for placements?",
        "Is there a placement cell?",
        "What is the average salary offered during placements?"
    ],
    "events": [
        "Are there any upcoming events?",
        "When is the next college fest?",
        "Does the college organize workshops?",
        "Are there any guest lectures or seminars?",
        "How can I participate in college events?"
    ],
    "faculty-CSD": [
        "Who are the faculty members of CSD?",
        "Can you tell me the professors in CSD?",
        "Who teaches in the CSD department?",
        "List the CSD faculty members.",
        "Faculty of CSD department?",
        "Tell me about CSD faculty.",

    ],
    "faculty-AI": [
        "Who are the faculty members of AI?",
        "Can you tell me the professors in AI?",
        "Who teaches in the AI department?",
        "List the AI faculty members.",
        "Faculty of AI department?",
        "Tell me about AI faculty."
    ],
    "faculty-civil": [
        "Who are the faculty members of civil?",
        "Can you tell me the professors in civil?",
        "Who teaches in the civil department?",
        "List the civil faculty members.",
        "Faculty of civil department?",
        "Tell me about civil faculty."
    ],
        "faculty-mechanical": [
        "Who are the faculty members of mechanical ?",
        "Can you tell me the professors in mechanical department?",
        "Who teaches in the mechanical department?",
        "List the mechanical faculty members.",
        "Faculty of mechanical department?",
        "Tell me about mechanical faculty."
    ],



}

responses = {
    "Hi": ["Hello, I am your chatbot. How may I assist you today?"],
    "admission": "You can apply online through our portal. The deadline is June 30. Make sure to check the eligibility criteria before applying.",
    "courses": "We offer a variety of courses including Computer Science, Business, and Arts. You can check the detailed syllabus on our website.",
    "fees": "The tuition fee varies by program. Please visit our finance section for detailed fee information and available scholarships.",
    "exam": "The next exam schedule will be published soon. You can register for exams through the student portal.",
    "HOD-AI department": "The Head of the AI department is Professor Dinesh Prajapati.",
    "HOD-IT department": "The Head of the IT department is Professor Narendra Chauhan.",
    "who": "I am an AI-powered chatbot created for ADIT College, designed by the students of AI department Deep shreyaskar and Dhruval Chauhan, how can I help you?",
    "time": "College starts at 9:00 AM and ends at 5:00 PM. There will be a lunch break from 12:30 PM to 1:30 PM.",
    "location": "ADIT College is located in the heart of the city at 123 College Street, City XYZ.",
    "facilities": "Yes,Our campus includes a modern library, sports facilities, hostels, a cafeteria, and Wi-Fi throughout the campus.",
    "internships": "Yes, we have a dedicated internship cell that connects students with companies offering internship opportunities. Most internships are paid, and the college provides placement support.",
    "placements": "We have a strong placement record, with top companies like Microsoft, Google, and IBM visiting our campus. The average salary during placements is $40,000 per year.",
    "events": "We host multiple events throughout the year, including fests, workshops, and guest lectures. Stay tuned for announcements on the website.",
    "exam_schedule": "You can find the detailed exam schedule on the student portal under the 'Exams' section.",
    "who_helps_in_internships": "Our Career Services team assists students in applying for internships and preparing for interviews.",
    "course_duration": "The duration of most courses is 4 years for undergraduate programs, and 2 years for postgraduate programs.",
    "scholarships": "Yes, we offer several scholarships based on merit and financial need. Please visit our finance section for more details.",
    "online_courses": "We do offer a few online courses. You can check out the available online programs on our website.",
    "working_hours": "The college operates from Monday to Friday, from 9:00 AM to 5:00 PM. Some labs may have extended hours for practical sessions.",
    "hostels": "Yes, we provide hostel facilities for both male and female students. Rooms are equipped with necessary amenities and 24/7 security.",
    "cafeteria": "Our cafeteria serves a variety of meals, including vegetarian and non-vegetarian options, along with snacks and beverages.",
    "wi_fi": "Yes, the campus is equipped with high-speed Wi-Fi, accessible to all students in the classrooms, library, and hostels.",
    "placement_assistance": "Our Placement Cell offers training sessions, mock interviews, and connects students with companies that visit the campus for placements.",
    "placement_record": "Our college boasts an excellent placement record, with many students securing positions in top multinational companies.",
    "placement_salary": "The average salary offered during placements is $45,000 annually, with some students earning even higher packages.",
    "upcoming_events": "Our next event is the Annual Tech Fest, scheduled for the first week of December. Stay tuned for more details.",
    "guest_lectures": "We frequently organize guest lectures by industry professionals. The next lecture will be on 'AI in Modern Technology' by Dr. Ravi Kumar.",
    "faculty-AI":[
        "Dr.Dinesh J Prajapati",
        "Prof.Anjali Rajput",
        "Prof.Mayur Ajmeri",
    ],
    "faculty-CSD":[
        "Prof.Khushali Patel",
        "Prof.Jitiksha Patel"
    ],
    "faculty-civil":[
        "Dr.Rajiv Bhatt",
        "Prof.Yagnik Darshan M.",
        "Dr.Drashti K.Bhatt",
        "Prof.Bhavin V. Patel",
        "Prof.Axay Shah",
        "Prof.Chhaya Bhrambhatt",
    ],
    "faculty-mechanical":[
        "Dr.Vishal N. Singh",
        "Dr.Yashwant D. Patel",
        "Dr.Mitesh I. Shah",
        "Dr.Ayanesh Y. Joshi",
        "Dr.Ronak R. Shah",
        "Dr.Manisha V. Makwana",
        "Prof.Bhaumik J. Sheth",
        "Prof.Sankalp B. Bhatia",
        "Prof.Mehul B. Patel",
        "Prof.Krunal J. Shah",
        "Prof.Harsh B. Joshi",
        "Prof.Tejas R. Prajapati",
        "Prof.Ruchir J. Desai",
        "Prof.Maharshi H. Thakkar",
        "Prof.Vaishal J. Banker",
        "Prof.Jagruti Jadav",
        "Prof.Bhavesh Parmar",

    ]

}


# Preprocess data for training
X, y = [], []
for intent, phrases in data.items():
    for phrase in phrases:
        X.append(phrase)
        y.append(intent)

# Train model using Naive Bayes
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(X, y)

# Save the trained model to a file
joblib.dump(model, "ml_model.pkl")


def get_response(user_input):
    try:
        model = joblib.load("ml_model.pkl")  # Load trained model
        intent = model.predict([user_input])[0]  # Predict intent

        response = responses.get(intent, "Sorry, I don't understand. Please contact support.")

        # ✅ Check if the intent is specifically for faculty listing
        faculty_intents = {"faculty-CSD", "faculty-AI", "faculty-civil","faculty-mechanical"}  # Add all faculty-related intent keys here

        if intent in faculty_intents and isinstance(response, list):
            numbered_response = "<br>".join(f"{i+1}. {faculty}" for i, faculty in enumerate(response))
            return f"Here are the faculty members:<br>{numbered_response}"

        return response  # Return normal string response for greetings or general queries

    except Exception as e:
        return f"Error processing request: {str(e)}"




# Function to generate a response using OpenAI API
# def generate_openai_response(user_input):
#     # Get the API key from environment variables (make sure it's set correctly)
#     openai.api_key = os.getenv("sk-...U2UA")
#
#     try:
#         # Make a request to OpenAI's GPT-3 model (or another model of your choice)
#         response = openai.Completion.create(
#             engine="text-davinci-003",  # or "gpt-4" if you have access to it
#             prompt=user_input,
#             max_tokens=150,  # Limit the length of the response
#             temperature=0.7,  # Control the randomness of the output
#         )
#         # Extract the response text
#         return response.choices[0].text.strip()
#
#     except Exception as e:
#         return f"Sorry, I encountered an error while processing your request. Please try again later. Error: {str(e)}"
#
#
# Example Usage
if __name__ == "__main__":
    user_input = input("You: ")
    response = get_response(user_input)
    print(f"Bot: {response}")

