import imageio as imageio
import numpy
import scipy.special
import matplotlib.pyplot
import scipy.ndimage
from PIL import Image


class neuralNetwork:

    # инициализировать нейронную сеть
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # задать количество узлов в входном, скрытом, выходном слоях
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # задать коэффициент обучения
        self.lr = learningrate

        # задание весовых коэффициентов между входным и скрытым узлами (wih) и скрытым и выходным (who)
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # использование сигмоиды в качестве функции активации
        self.activation_function = lambda x: scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)
        # pass

    # тренировка нейронной сети
    def train(self, inputs_list, targets_list):
        # преобразовать список входных значений в двухмерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # рассчитать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        # ошибка = целевое значение - фактическое значение
        output_errors = targets - final_outputs

        # ошибки скрытого слоя - это ошибки output_errors,
        # распределенные пропорционально весовым коэффициентам связей
        # и рекомбинированные на скрытых узлах
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # обновить весовые коэффициенты связей между скрытым и выходным слоями
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))

        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))
        pass

    # опрос нейронной сети
    def query(self, inputs_list):
        # преобразовать список входных значений
        # в двухмерный массив
        inputs = numpy.array(inputs_list, ndmin=2).T

        # расситать входящие сигналы для скрытого слоя
        hidden_inputs = numpy.dot(self.wih, inputs)
        # рассчитать исходящие сигналы для скрытого слоя
        hidden_outputs = self.activation_function(hidden_inputs)

        # рассчитать входящие сигналы для выходного слоя
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # рассчитать исходящие сигналы для выходного слоя
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    # backquery the neural network
    # we'll use the same termnimology to each item,
    # eg target are the values at the right of the network, albeit used as input
    # eg hidden_output is the signal to the right of the middle nodes
    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = numpy.array(targets_list, ndmin=2).T

        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = numpy.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)

        # calculate the signal out of the input layer
        inputs = numpy.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs


def resize_image(input_image_path,
                output_image_path,
                size):
    original_image = Image.open(input_image_path)
    width, height = original_image.size
    print('The original image size is {wide} wide x {height} '
              'high'.format(wide=width, height=height))

    resized_image = original_image.resize(size)
    width, height = resized_image.size
    print('The resized image size is {wide} wide x {height} '
              'high'.format(wide=width, height=height))
    # resized_image.show()
    resized_image.save(output_image_path)


if __name__ == "__main__":
    # задание начальных значений и запуск сети
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10

    # коэффициент обучения равен 0.3
    learning_rate = 0.1

    # создать экземпляр нейронной сети
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

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

    # берем записи для проверки работы сети
    test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    # тестирование нейронной сети

    # журнал оценок работы сети, первоначально пустой
    scorecard = []

    # перебрать все записи в тестовом наборе данных

    for record in test_data_list:
        # получить список значений из записи, используя символы
        # запятой (',') в качестве разделителя
        all_values = record.split(',')
        # правильный ответ - первое значение
        correct_label = int(all_values[0])
        # print(correct_label, " истинный маркер")
        # масштабировать и сместить входные значения
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # опрос сети
        outputs = n.query(inputs)
        # индекс наибольшего значения является маркерным значением
        label = numpy.argmax(outputs)
        # print(label, " ответ сети")
        # присоединить оценку ответа сети к концу списка
        if label == correct_label:
            # в случае правильного ответа сети присоединить к списку значение 1
            scorecard.append(1)
        else:
            # в случае неправильного ответа сети присоединить к списку значение 0
            scorecard.append(0)
            pass

    # рассчитать показатель эффективности в виде доли правильных ответов
    scorecard_array = numpy.asarray(scorecard)
    print("эффективность = ", scorecard_array.sum() / scorecard_array.size)

    angle = 45

    # запускаем сигналы по сети в обратную сторону, чтобы посмотреть,
    # как представляются для сети данные (как она видит числа)

    # маркер для проверки
    # label = 9
    # создаем выходной массив для данного маркера
    # targets = numpy.zeros(output_nodes) + 0.01
    # all_values[label] назначаем правильным ответом
    # targets[label] = 0.99
    # print(targets)

    # получаем данные для отображения
    # image_data = n.backquery(targets)

    # строим изображение
    # matplotlib.pyplot.imshow(image_data.reshape(28, 28), cmap='Greys', interpolation='None')
    # matplotlib.pyplot.show()
