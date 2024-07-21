from django.shortcuts import render, redirect
from .models import User
from django.conf import settings
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

model_path = os.path.join(settings.BASE_DIR, 'D:/Github Programs/Projects/RealEye/RealEyeUI/realeye/Models/realeyemodel2.pt')

model = ViTForImageClassification.from_pretrained(model_path, torchscript=True)
processor = ViTImageProcessor.from_pretrained("dima806/ai_vs_real_image_detection")

transform = Compose([
    Resize((224, 224)),  
    ToTensor(),
    Normalize(mean=processor.image_mean, std=processor.image_std)
])

def classify_image(image):
   
    image = transform(image)
    image = image.unsqueeze(0)  
    
    model.eval()
    with torch.no_grad():
        outputs = model(pixel_values=image)
        
       
        if isinstance(outputs, tuple):
          
            logits = outputs[0]
        else:
            
            logits = outputs.logits
        
       
        predicted_probs = torch.softmax(logits, dim=-1)
        predicted_class_idx = torch.argmax(predicted_probs, dim=-1).item()
        predicted_class_label = model.config.id2label[predicted_class_idx]
    
    return predicted_class_label


def login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        user = User.objects.create(username=username, email=email)
        return redirect('upload')
    
    return render(request, 'login.html')

def upload(request):
    if request.method == 'POST' and request.FILES['image']:
        image_file = request.FILES['image']
        
        image_path = os.path.join(settings.MEDIA_ROOT, image_file.name)
        with open(image_path, 'wb+') as destination:
            for chunk in image_file.chunks():
                destination.write(chunk)
        
        image = Image.open(image_path).convert("RGB")
        result = classify_image(image)
        
        os.remove(image_path)  
        
        return redirect('result', result=result)
    
    return render(request, 'upload.html')

def result(request, result):
    display_result = "AI-Generated" if result == "FAKE" else "Real"
    return render(request, 'result.html', {'display_result': display_result})

