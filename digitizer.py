import imageio
import numpy
import scipy
import telebot
import neural_network as nn
from PIL import Image, ImageEnhance

# создаем бота и даем ему токен авторизации
bot = telebot.TeleBot('{INPUT YOUR BOT KEY}')

# тренируем нейронную сеть перед началом работы бота
# задание начальных значений и запуск сети
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# коэффициент обучения равен 0.3
learning_rate = 0.1

# создать экземпляр нейронной сети
n = nn.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# загрузить в список тестовый набор данных
training_data_file = open("mnist_dataset/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

# тренировка нейронной сети

# пееременная epochs указывает, сколько раз тренировочный набор
# данныъ используется для треировки сети
epochs = 5

# перебрать все записи в треировочном наборе данных
for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')
        # нормирование значений в диапазон 0.01 - 1.0
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01

        # all_values[0] - целевое маркерное значение для данной записи
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

        # rotated clockwise by -x degrees
        inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), 10, cval=0.01, order=1,
                                                              reshape=False)
        n.train(inputs_plusx_img.reshape(784), targets)
        # rotated clockwise by x degrees
        inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28, 28), -10, cval=0.01, order=1,
                                                               reshape=False)
        n.train(inputs_minusx_img.reshape(784), targets)
        # rotated anticlockwise by 10 degrees
        # inputs_plus10_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01, order=1, reshape=False)\n",
        # n.train(inputs_plus10_img.reshape(784), targets)\n",
        # rotated clockwise by 10 degrees\n",
        # inputs_minus10_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), -10, cval=0.01, order=1, reshape=False)\n",
        # n.train(inputs_minus10_img.reshape(784), targets)\n",
        pass
    pass
print('Training is complete')

start_menu = telebot.types.ReplyKeyboardMarkup()
start_menu.row('Отправить фото', 'Посмотреть образы')
start_menu.row('Обо мне')

digits = telebot.types.ReplyKeyboardMarkup()
digits.row('0')
digits.row('1', '2', '3')
digits.row('4', '5', '6')
digits.row('7', '8', '9')
digits.row('Назад')

# напишем декоратор, чтобы бот отвечал на сообщение /start
@bot.message_handler(commands=['start'])
def start_message(message):
    print("Writing answer to start message")
    bot.send_message(message.chat.id, 'Стартовое меню. Выберите раздел', reply_markup=start_menu)


@bot.message_handler(content_types=['photo'])
def send_photo(message):
    print("Handling photo")
    raw = message.photo[0].file_id
    name = 'photos/' + raw + ".jpg"
    file_info = bot.get_file(raw)
    downloaded_file = bot.download_file(file_info.file_path)

    with open(name, 'wb') as new_file:
        new_file.write(downloaded_file)
    bot.reply_to(message, "Фото обрабатывается")
    img = Image.open(name)
    img = ImageEnhance.Brightness(img)
    bw = img.enhance(2.0)
    bw.save(name)
    nn.resize_image(input_image_path=name,
                    output_image_path=name,
                    size=(28, 28))
    # угол, на который мы повораичваем фото, чтобы избавиться от шумов в углах
    angle = 45

    # проверяем сделанные нами фотографии
    # откроем и преобразуем наше фото
    img_array = imageio.imread(name, as_gray=True)
    img_array = scipy.ndimage.interpolation.rotate(img_array, angle, cval=255, order=1,
                                                   reshape=False)
    img_array = scipy.ndimage.interpolation.rotate(img_array, -angle, cval=255, order=1,
                                                   reshape=False)

    img_data = 255.0 - img_array.reshape(784)
    img_data = (img_data / 255.0 * 0.99) + 0.01

    # четырые фотографии, четыре ответа
    outputs = n.query(img_data)
    # индекс наибольшего значения является маркерным значением
    label = numpy.argmax(outputs)
    bot.reply_to(message, "На фото изображена цифра {label}".format(label=label))


# напишем декоратор, чтобы отвечать на текстовые сообщения
@bot.message_handler(content_types=['text'])
def send_text(message):
    print("answering text message")
    if message.text.lower() == 'отправить фото':
        bot.send_message(message.chat.id, 'Приложите фото в формате квадрат (1x1), на котором на белом фоне '
                                          'изображена любая цифра')
    elif message.text.lower() == 'посмотреть образы':
        bot.send_message(message.chat.id, 'Выберите цифру, образ которой хотите посмотреть', reply_markup=digits)
    elif message.text.lower() == 'обо мне':
        bot.send_message(message.chat.id, 'Привет! Я бот Digitizer. Моя задача - распознавать цифры, которые ты мне '
                                          'отправляешь. Для того, чтобы уметь делать это, я использую простую '
                                          'трехслойную нейронную сеть. Она обучена на наборе рукописных цифр MNIST. '
                                          'Чтобы я корректно распознал отправленную мне цифру, она должна быть '
                                          'написана на белом фоне, в фотографии формата "квадрат". Цвет, которым '
                                          'написана цифра должен быть темным.\n\nЭтот проект будет развиваться и '
                                          'дальше. Скоро появится визуализация работы сети, а после - распознавание '
                                          'не только цифр, но и других объектов на фотографиях.\nСпасибо за интерес!!!',
                         reply_markup=start_menu)
    elif message.text.lower() == 'назад':
        print("Writing answer to start message")
        bot.send_message(message.chat.id, 'Стартовое меню. Выберите раздел', reply_markup=start_menu)
    elif message.text.lower() == '0':
        photo = open('D:\python_projects\digitizer\\neuro_photos\zero.png', 'rb')
        bot.send_photo(message.chat.id, photo)
        photo.close()
    elif message.text.lower() == '1':
        photo = open('D:\python_projects\digitizer\\neuro_photos\one.png', 'rb')
        bot.send_photo(message.chat.id, photo)
        photo.close()
    elif message.text.lower() == '2':
        photo = open('D:\python_projects\digitizer\\neuro_photos\\two.png', 'rb')
        bot.send_photo(message.chat.id, photo)
        photo.close()
    elif message.text.lower() == '3':
        photo = open('D:\python_projects\digitizer\\neuro_photos\\three.png', 'rb')
        bot.send_photo(message.chat.id, photo)
        photo.close()
    elif message.text.lower() == '4':
        photo = open('D:\python_projects\digitizer\\neuro_photos\\four.png', 'rb')
        bot.send_photo(message.chat.id, photo)
        photo.close()
    elif message.text.lower() == '5':
        photo = open('D:\python_projects\digitizer\\neuro_photos\\five.png', 'rb')
        bot.send_photo(message.chat.id, photo)
        photo.close()
    elif message.text.lower() == '6':
        photo = open('D:\python_projects\digitizer\\neuro_photos\six.png', 'rb')
        bot.send_photo(message.chat.id, photo)
        photo.close()
    elif message.text.lower() == '7':
        photo = open('D:\python_projects\digitizer\\neuro_photos\seven.png', 'rb')
        bot.send_photo(message.chat.id, photo)
        photo.close()
    elif message.text.lower() == '8':
        photo = open('D:\python_projects\digitizer\\neuro_photos\eight.png', 'rb')
        bot.send_photo(message.chat.id, photo)
        photo.close()
    elif message.text.lower() == '9':
        photo = open('D:\python_projects\digitizer\\neuro_photos\\nine.png', 'rb')
        bot.send_photo(message.chat.id, photo)
        photo.close()


bot.polling()
