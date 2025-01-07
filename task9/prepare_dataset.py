import os
from PIL import Image, ImageFilter

input_folder = "celeba/"
output_folder = "celeba_2_16/"

blur_images = False

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

counter = 0

for filename in os.listdir(input_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        counter = counter + 1
        input_path = os.path.join(input_folder, filename)

        with Image.open(input_path) as img:
            img = img.resize((90, 110))

            # Замилюємо зображення, якщо параметр blur_images True
            if blur_images:
                img = img.filter(ImageFilter.GaussianBlur(2))

            # Виражаємо центральну частину розміром 64x64
            width, height = img.size
            left = (width - 64) // 2
            upper = (height - 64) // 2
            right = (width + 64) // 2
            lower = (height + 64) // 2

            img_cropped = img.crop((left, upper, right, lower))
            img_cropped = img_cropped.resize((16, 16))

            # Шлях для збереження обробленого зображення
            output_path = os.path.join(output_folder, filename)

            # Зберігаємо вирізане зображення
            img_cropped.save(output_path)
        print(counter)
    if counter == 10000:
        break

print("Обробка зображень завершена!")
