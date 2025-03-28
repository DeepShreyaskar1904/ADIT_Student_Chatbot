from django.urls import path
from rest_framework.decorators import api_view

from .views import chatbot_response,chat_page

urlpatterns = [
    path("", chat_page, name="chat_page"),
    path("api/chatbot/", chatbot_response, name="chatbot_response"),
]

