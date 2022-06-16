import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from vit_model import vit_base_patch16_224 as create_model


def main():

    races = ['WHITE', 'BLACK', 'ASIAN', 'HISPANIC', 'OTHER']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_high_mean = [0.1876, 0.1876, 0.1876]
    train_high_std = [0.2800, 0.2800, 0.2800]
    train_low_mean = [0.2077, 0.2077, 0.2077]
    train_low_std = [0.2873, 0.2873, 0.2873]

    '''
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    '''

    transform_test_high = transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(train_high_mean, train_high_std)
        ])

    transform_test_low = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(train_low_mean, train_low_std)
    ])

    '''
    # load image
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    '''

    img_path = './2542.other.png'
    img = Image.open(img_path)
    # plt.imshow(img)
    # plt.show()
    img_high = transform_test_high(img).unsqueeze(0)
    img_low = transform_test_low(img).unsqueeze(0)

    img_high = img_high.to(device)
    img_low = img_low.to(device)

    '''
    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)
    '''

    # create model
    model_high = create_model(num_classes=3).to(device)
    model_low = create_model(num_classes = 3).to(device)
    # load model weights
    model_high_weight_path = "./weights/model-4_high_904.pth"
    model_low_weight_path = "./weights/model-12_low_729.pth"
    model_high.load_state_dict(torch.load(model_high_weight_path, map_location=device))
    model_low.load_state_dict(torch.load(model_low_weight_path, map_location = device))
    model_high.eval()
    model_low.eval()

    with torch.no_grad():
        # predict class
        output = model_high(img_high)
        pred_classes = torch.max(output, dim=1)[1]

        if pred_classes == 2:
            output = model_low(img_low)
            pred_classes = torch.max(output, dim=1)[1]
            if pred_classes == 0:
                pred_classes = 2
            elif pred_classes == 1:
                pred_classes = 3
            elif pred_classes == 2:
                pred_classes = 4

    race = races[pred_classes]
    race = 'Predicted results: ' + race
    plt.imshow(img)
    plt.text(20, 20, race, color = 'red')
    plt.show()
    print(race)

    '''
    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()
    '''

if __name__ == '__main__':
    main()
