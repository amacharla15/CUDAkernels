import torch
import torchvision
import torch.nn as nn

#converting images to tensors
transform=torchvision.transforms.ToTensor()


#mnist has already inbuilt train param
train_dataset=torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset=torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

#batching
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)



class MLP(nn.Module):

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)   
        self.layer2 = nn.Linear(128, 10) 

    def forward(self,x):
        x = x.view(-1, 784)         # [64, 1, 28, 28] → [64, 784] (flatten)
        x = self.layer1(x)          # [64, 784] → [64, 128] (weights × input + bias)
        x = torch.relu(x)           # [64, 128] → [64, 128] (negatives become 0)
        x = self.layer2(x)          # [64, 128] → [64, 10] (10 scores per image)
        return x


if __name__ == "__main__":
    model=MLP()

    #training
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(0,10):
        for images, labels in train_loader:
            pred=model(images)
            loss=loss_fn(pred,labels)
            optimizer.zero_grad()  
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    #testing
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            pred = model(images)
            predicted_digits = torch.argmax(pred,dim=1)
            a=(predicted_digits==labels).sum()
            correct += a
            total += labels.size(0)
    accuracy = correct / total
    print(accuracy)