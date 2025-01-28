import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import argparse


from network import P_Net
from data_purify import Data_Purify



# CHECK GPU/CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(f"Using device: {device}")


# Parse the command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--theta', type=float, default=1)
parser.add_argument('--dataset',type=str,default='drone_1_z.csv', help='corresponding theta dataset')
parser.add_argument('--lr',type=float, default=0.001, help='learning rate')
parser.add_argument('--thresholdp',type=int,default= 200, help='filter outliters of p')
parser.add_argument('--thresholdu',type=int,default=3,help='filter outliters of u')
parser.add_argument('--pre_trained', type=str, default='weights_drone/1_z/drone_theta1_epoch900_19.738344_0.000232.pth',help='input your pretrained weight path if you want')
parser.add_argument('--onereference',type=bool,default=False,help='determine if you there is only a reference')

args = parser.parse_args()

print(args)

theta = args.theta
save_path = f'weights_drone/{theta}_z'
if not os.path.exists(save_path):
    os.mkdir(save_path)

# import dataset and purify outliers    
data = Data_Purify(args.dataset,args.thresholdp,args.thresholdu)

# THERE ARE SOME REFERENCE THERSHOLDS FOR DIFFERENT DATASET
# data = Data_Purify('data_01theta_ref_new.csv',10,10)
# data = Data_Purify('data_1theta_ref_new.csv',40,15)

if not args.onereference:
    input_data_combine, label_data = data.purified_data()
else:
    print('part of dataset is selected!')
    _, _ = data.purified_data()
    # IMPORTANT: CHOOSE REFERENCE IF YOU SWITCH TO THIS
    # input_data_combine, label_data = data.one_reference(0)
    input_data_combine, label_data = data.ROI_data()
# data.draw_data()


X_train, X_test, y_train, y_test = train_test_split(input_data_combine, label_data, test_size=0.2, random_state=42)

X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.Tensor(y_train)
X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.Tensor(y_test)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

batch_size = 1024
num_epochs = 5000


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)





model = P_Net(output_size=4).to(device)
start_epoch = 0

if len(args.pre_trained):
    model.load_state_dict(torch.load(args.pre_trained))
    model.eval()
    print('----------added previous weights: {}------------'.format(args.pre_trained))
    start_epoch = int(args.pre_trained.split('/')[-1].split('_')[-3].split('epoch')[-1])

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# Define the cosine annealing scheduler
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=0.0005)



# test saved model
# # Create an instance of the model
# model = FullyConnectedLayer(input_size, hidden_size, output_size)
# # Load the saved model's state dictionary
# model.load_state_dict(torch.load('trained_model_0.2641225345145178.pth'))

total_loss = []
loss_u_total = []
def test(model, test_loader,epoch):
    test_loss = 0.0
    model.eval()
    loss_set = []
    with torch.no_grad():
        loop = tqdm(test_loader)
        for inputs, targets in loop:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            u_loss = criterion(outputs[:,3], targets[:,3].to(device))
            # print(outputs)
            # print(targets)
            # print(loss)
            if not 'cuda' in device.type:
                loss_set.append(loss.item())
                test_loss += loss.item() * inputs.size(0)
                u_loss += np.average(u_loss.item())
            else:
                loss_set.append(loss.cpu().item())
                test_loss += loss.cpu().item() * inputs.size(0)
                u_loss += np.average(u_loss.cpu().item())

            loop.set_postfix(loss=f"{loss_set[-1]:.4f}", refresh=True)
    test_loss /= len(test_loader.dataset)
    u_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}, u loss: {u_loss:.4f}')

    # save loss list
    total_loss.append(test_loss)
    loss_u_total.append(u_loss)

    # Save the model
    if len(total_loss)==1:
        return 0

    if (test_loss <= min(total_loss) or u_loss <= min(loss_u_total)) and epoch != 0:
        print('Model Saved!')
        torch.save(model.state_dict(), os.path.join(save_path,'drone_theta{}_epoch{}_{:3f}_{:3f}.pth'.format(theta,epoch,np.average(loss_set),1e3*u_loss)))




test(model, test_loader,epoch=0)


losses = []
loss_avg = []
model.train()
for epoch in range(start_epoch, num_epochs):
    for inputs, targets in train_loader:
        # clear the gradient for each mini batch.
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss1 = 10*criterion(outputs[:,:3], targets[:,:3].to(device))
        loss2 = 10000*criterion(outputs[:,3], targets[:,3].to(device))
        # loss1 = criterion(outputs[:,:3], targets[:,:3].to(device))
        # loss2 = 10*criterion(outputs[:,3], targets[:,3].to(device))
        loss = loss1 + loss2
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=100)

        # Step the scheduler
        scheduler.step()
        # Update the model parameters
        optimizer.step()

        loss_avg.append(loss.item())  # Store the loss value for plotting

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    losses.append(np.average(loss_avg))

    if epoch >= 400 and epoch%5==0:
        test(model, test_loader,epoch)

# Plot the loss dynamically
plt.clf()  # Clear previous plot
plt.plot(losses, label='Training Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
# plt.pause(0.05)  # Pause for a short time to update the plot
plt.savefig(os.path.join(save_path,'training_loss_{}.png'.format(num_epochs)))
plt.plot()




