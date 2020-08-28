import imageio as imageio
import matplotlib.pyplot
from PIL import Image, ImageEnhance


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


if __name__ == '__main__':
    # img = Image.open('test_photo/IMG_0807.jpg')
    # img_array1 = imageio.imread('test_photo/final_size.jpg', as_gray=True)
    # gray = img.convert('L')
    # bw = gray.point(lambda x: 0 if x < 128 else x, '1')
    # img = ImageEnhance.Brightness(img)
    # bw = img.enhance(2.0)
    # bw.save('test_photo/final_size_3.jpg')
    # matplotlib.pyplot.imshow(bw, cmap='Greys', interpolation='None')
    # matplotlib.pyplot.show()

    resize_image(input_image_path='test_photo/final_size_3.jpg',
                 output_image_path='test_photo/final_size_3.jpg',
                 size=(28, 28))


