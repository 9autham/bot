import telebot
import requests
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

# Replace 'YOUR_BOT_TOKEN' with your actual bot token
bot_token = '6672522308:AAE2T_3N-1G-q3jBXFRj2ZDsieN3mFoCTgA'

# Telegram Bot Instance
bot = telebot.TeleBot(bot_token)
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../potatoes.h5")
MODEL = tf.keras.models.load_model(model_path)
CLASS_NAMES = ['Early Blight', 'Late Blight', 'Healthy']

# Handle the /start command
@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(message.chat.id, "Start Cheyee bey")

# Handle incoming images
@bot.message_handler(content_types=['photo'])
def handle_image(message):
    try:
        # Get the file ID of the largest available photo
        file_id = message.photo[-1].file_id
        file_info = bot.get_file(file_id)

        # Download the image from Telegram servers
        image_url = f'https://api.telegram.org/file/bot{bot_token}/{file_info.file_path}'
        response = requests.get(image_url)
        image = np.array(Image.open(BytesIO(response.content)).resize((256, 256)))
        img_batch = np.expand_dims(image, 0)
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        # Set a confidence threshold
        confidence_threshold = 0.1

        if confidence >= confidence_threshold:
            # Valid leaf image
            bot.send_message(message.chat.id, f"Predicted class: {predicted_class}\nConfidence: {confidence}")
        else:
            # Not a leaf image
            bot.send_message(message.chat.id, "This doesn't seem to be a leaf image.")

    except Exception as e:
        # Handle any errors that may occur
        print(str(e))
        bot.reply_to(message, "Oops! Something went wrong.")

# Handle any other text messages
@bot.message_handler(func=lambda message: True)
def echo_message(message):
    bot.reply_to(message, "I can only handle images. Please send me a plant image.")

# Start the bot
if __name__ == "__main__":
    bot.polling()
