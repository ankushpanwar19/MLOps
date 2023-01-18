"""
Trainer script
"""
import torch
import torch.nn.functional as F

class MnistTrainer():
    """
    Model trainer on MNIST dataset
    """
    def __init__(self, model, device='cpu') -> None:
        self.model = model
        self.device = device

    def train(self, args, train_loader, optimizer, epoch):
        """
        Function to train the model
        """
        self.model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print(f'Train Epoch: {epoch} \
                    [{batch_idx*len(data)}/{len(train_loader.dataset)} \
                    ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                if args.dry_run:
                    break

    def test(self, test_loader):
        """
        Testing models
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy =  100. * correct / len(test_loader.dataset)
        print(f'\nTest set: Average loss: {test_loss:.4f}, \
                Accuracy: {correct}/{len(test_loader.dataset)} \
                ({ 100. * correct / len(test_loader.dataset):.0f}%)\n')
        
        return test_loss, test_accuracy


if __name__ == "__main__":

    from src.models.custom_model import MnistNet
    mnistmodel = MnistNet()
    trainer = MnistTrainer(mnistmodel,device='cpu')
    print("end")
