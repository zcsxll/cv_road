import os
import sys
sys.path.append(os.path.abspath('.'))
import torch

from transform import create_transform
from dataset import create_dataset
from net import create_model

if __name__ == '__main__':
    transform = create_transform('transform1')()
    dataset = create_dataset('dataset1')(transform)
    data_loader = torch.utils.data.DataLoader(
        dataset = dataset,
        batch_size = 24,
        shuffle = False,
        num_workers = 24)

    model = create_model('u-net')(3, 1)
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids = [i for i in range(8)])

    optimizer = torch.optim.Adam(model.parameters(), 0.001)

    # log = SummaryWriter('./logs/loss_1epoch')
    print("totally %d steps" % len(data_loader))
    for step, (batch_x, batch_y) in enumerate(data_loader):
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()
        # print(batch_x.shape, batch_y.shape)

        output = model(batch_x).squeeze(dim=1)
        # print(output.shape)
        loss = torch.nn.MSELoss()(output, batch_y)
        print("step: %d, loss: %f" % (step, loss.cpu().detach().numpy()))
        # log.add_scalar("loss", loss.cpu().detach().numpy(), step)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if step % 1000:
        #     torch.save(model.state_dict(), "./model.pkl")
    torch.save(model.state_dict(), "./model.pkl")
    # log.close()