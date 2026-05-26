from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.ImageFolder(root='dataset/train', transform=transform)
test_set  = datasets.ImageFolder(root='dataset/test',  transform=transform)

print(train_set[0][0].shape)
print(train_set.classes)
print(train_set.class_to_idx)
print(len(train_set))
print(len(test_set))