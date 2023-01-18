import argparse
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import mlflow

import sys
# sys.path.insert(0,'src')
from training.trainer import MnistTrainer
from models.custom_model import MnistNet

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset_train = datasets.MNIST('./data/raw', train=True, download=True,
                       transform=transform)
    dataset_test = datasets.MNIST('./data/raw', train=False,
                       transform=transform)

    # reducing the size of train data for quick test
    filter_idx = list(range(1, len(dataset_train), 5))
    dataset_train = torch.utils.data.Subset(dataset_train, filter_idx)   

    train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

    model = MnistNet().to(device)
    trainer = MnistTrainer(model,device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    
    EXPERIMENT_NAME = "MNIST_mlflow_demo"

    existing_exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if not existing_exp:
        EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
    
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run( run_name="Hyperparameter_optimize") as run:
        for epoch in range(1, args.epochs + 1):
            trainer.train(args, train_loader, optimizer, epoch)
            test_loss, test_accuracy = trainer.test(test_loader)
            # Track metrics
            mlflow.log_metric("test_loss", test_loss,step=epoch)
            mlflow.log_metric("test_accuracy", test_accuracy,step=epoch)
            scheduler.step()

        
        # Track parameter
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("gamma", args.gamma)
        mlflow.log_param("epoch", args.epochs)

        # # Track metrics
        # mlflow.log_metric("test_loss", test_loss)
        # mlflow.log_metric("test_accuracy", test_accuracy)
        
        # Track model
        mlflow.pytorch.log_model(model, "classifier")

        # Save model artifact
        model_save_dir = "models/mnist_cnn.pt"
        torch.save(model.state_dict(), model_save_dir)
        mlflow.log_artifact(model_save_dir)


if __name__ == "__main__":
    main()
