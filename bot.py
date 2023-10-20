import telebot
import urllib


token = "6545009319:AAG1YRp7Cr_FOg2jwFBSXYrDR39dgpHoazM"
bot = telebot.TeleBot(token)

@bot.message_handler(content_types=["document"])
def get_doc_messages(message):
    document_id = message.document.file_id
    file_info = bot.get_file(document_id)
    urllib.request.urlretrieve(f'http://api.telegram.org/file/bot{token}/{file_info.file_path}', file_info.file_path)


bot.polling(none_stop=True, interval=0)