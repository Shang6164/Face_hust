import numpy as np
import torchvision

from model_architecture import model
import torch
from dataloader.loaddata import DataLoader

root = "C:/Users/daoda/OneDrive/Documents/TestImage"
# define model
net = model.ShuffleFaceNet()
checkpoint = torch.torch.load("100.ckpt", map_location="cpu")
net.load_state_dict(checkpoint["net_state_dict"])
net.eval()

testset = DataLoader(root=root, flip=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset),
                                         shuffle=False, drop_last=False)
data_ = DataLoader(root="C:/Users/daoda/OneDrive/Documents/Anh do an", flip=False)
dataset = torch.utils.data.DataLoader(data_, batch_size=len(data_), shuffle=False, drop_last=False)

# with net.zero_grad():
all_data_dict = []
test_data_dict = []

for data in dataset:
    img, label = data[0], data[1]
    with torch.no_grad():
        all_data_result = net(img)
    i = 0
    for vector in all_data_result:
        dict_ = {"name": data_.label_name[i], "embedding": torch.Tensor.tolist(vector)}
        all_data_dict.append(dict_)
        i += 1

for data in testloader:
    img, label = data[0], data[1]
    with torch.no_grad():
        test_data_result = net(img)
    i = 0
    label = torch.Tensor.tolist(label)
    for vector in test_data_result:
        dict_ = {"name": testset.label_name[i], "embedding": torch.Tensor.tolist(vector)}
        test_data_dict.append(dict_)
        i += 1

# print(all_data_dict)
# print(test_data_dict)
tn = 0
fn = 0

for i in test_data_dict:
    max_ = -1
    max_li = all_data_dict[0]
    a = i["embedding"]
    for j in all_data_dict:
        b = j["embedding"]
        cosine = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        if abs(cosine) > max_:
            max_ = abs(cosine)
            max_li = j

    if max_li["name"] == i["name"]:
        tn += 1
    else:
        fn += 1
    print(max_li["name"], i["name"])
print(f"Acc: {tn/(tn+fn)}")
# print(cos(all_data_dict[0]["embedding"], test_data_dict[0]["embedding"]))
