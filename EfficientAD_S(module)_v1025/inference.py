import torch
import os
from needmodule import ImageFolderWithPath, visualize_one_sample
from torchvision import transforms
import time
path = 'output/small_v1_10_25/trainings/mvtec_ad' #저장된 모델 불러오는 코드
auto = torch.load(path+'/autoencoder_final.pth')
stu = torch.load(path+'/student_final.pth')
tea = torch.load(path+'/teacher_final.pth')
data_path = './Data_test' # test하고 싶은 폴더명 지정하시면 됩니다. 하지만 폴더안에 폴더를 하나 더 만들고 이미지 파일 삽입하셔야합니다.
image_size = 256
default_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
if __name__ == '__main__':
    test_set = ImageFolderWithPath(
        os.path.join(data_path))
    list_a = []
    for image, _, _ in test_set:
        orig_width = image.width
        orig_height = image.height
        image = default_transform(image)
        image = image[None]
        image = image.cuda()
        start_time = time.time()
        a = visualize_one_sample(image, orig_height, orig_width, auto, stu, tea)
        end_time = time.time()  # 종료 시간 저장
        elapsed_time = end_time - start_time  # 처리에 걸린 시간
        fps = 1.0 / elapsed_time  # FPS 계산
        print(f"FPS: {fps}")
        print(a)