import PIL.Image as Image
import os
from torchvision import transforms as transforms

# pytorch提供的torchvision主要使用PIL的Image类进行处理，所以它数据增强函数大多数都是以PIL作为输入，并且以PIL作为输出。
# 读取图片
def read_PIL(image_path):
    image = Image.open(image_path)
    return image

# 获取读到图片的不带后缀的名称
def get_name(image):
    im_path = image.filename
    im_name = os.path.split(im_path)    # 将路径分解为路径中的文件名+扩展名，获取到的是一个数组格式，最后一个是文件名
    name = os.path.splitext(im_name[len(im_name) - 1])     # 获取不带扩展名的文件名，是数组的最后一个
    return name[0] # arr[0]是不带扩展名的文件名，arr[1]是扩展名

# 将图片Reszie
def resize_img(image):
    Resize = transforms.Resize(size=(224, 224))
    resize_img = Resize(image)
    return resize_img

##################################################################################################

# 读取图片
image = read_PIL('./tup/tenniscourt72.tif')
print(image.size)  # 输出原图像的尺寸
name = get_name(image) # 获取读到图片的不带后缀的名称

# 创建输出目录
outDir = './logs'
os.makedirs(outDir, exist_ok=True)



random_cropped_image = resize_img(image)  # 随机裁剪

random_cropped_image.show()  # 显示裁剪后的图片

out_name = name + '_crop_' + str(1) + '.png'
print(out_name)
random_cropped_image.save(os.path.join(outDir, out_name))  # 按照路径保存图片
