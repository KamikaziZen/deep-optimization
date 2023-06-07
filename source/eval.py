import torch


def eval_loss(model, dataloader, loss_function, device):
    model.eval()
    loss = 0
    with torch.no_grad():
        for x, _ in dataloader:

            x = x.to(device)
            z = model(x)

            batch_loss = loss_function(z, x)

            loss += batch_loss.item()
            
    return loss / len(dataloader)