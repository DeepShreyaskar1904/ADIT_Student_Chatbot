from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .ml_model import get_response
from django.shortcuts import render

@csrf_exempt
def chatbot_response(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            user_message = data.get("message", "")
            bot_reply = get_response(user_message)
            return JsonResponse({"reply": bot_reply})
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)
@csrf_exempt

def chat_page(request):
    return render(request, "index.html")