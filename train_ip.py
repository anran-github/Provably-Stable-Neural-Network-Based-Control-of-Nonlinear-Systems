import pandas as pd
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
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
parser.add_argument('--theta', type=float, default=0.01)
parser.add_argument('--dataset',type=str,default='IP_roi_001theta_ref.csv', help='corresponding theta dataset')
parser.add_argument('--lr',type=float, default=0.001, help='learning rate')
parser.add_argument('--thresholdp',type=int,default= 20, help='filter outliters of p')
parser.add_argument('--thresholdu',type=int,default=3,help='filter outliters of u')
parser.add_argument('--pre_trained', type=str, default='',help='input your pretrained weight path if you want')
parser.add_argument('--onereference',type=bool,default=False,help='determine if you there is only a reference')

args = parser.parse_args()

print(args)

theta = args.theta
save_path = f'weights_roi/{theta}'
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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)





model = P_Net(output_size=4).to(device)

if len(args.pre_trained):
    model.load_state_dict(torch.load(args.pre_trained))
    model.eval()
    print('----------added previous weights: {}------------'.format(args.pre_trained))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)




# test saved model
# # Create an instance of the model
# model = FullyConnectedLayer(input_size, hidden_size, output_size)
# # Load the saved model's state dictionary
# model.load_state_dict(torch.load('trained_model_0.2641225345145178.pth'))

init_test_loss = []

def test(model, test_loader,epoch):
    test_loss = 0.0
    model.eval()
    loss_set = []
    with torch.no_grad():
        loop = tqdm(test_loader)
        for inputs, targets in loop:
            outputs = model(inputs.to(device))
            loss = criterion(outputs, targets.to(device))
            # print(outputs)
            # print(targets)
            # print(loss)
            if not 'cuda' in device.type:
                loss_set.append(loss.item())
                test_loss += loss.item() * inputs.size(0)
            else:
                loss_set.append(loss.cpu().item())
                test_loss += loss.cpu().item() * inputs.size(0)

            loop.set_postfix(loss=f"{loss_set[-1]:.4f}", refresh=True)
    test_loss /= len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')

    # save model
    # Save the model
    if len(init_test_loss)==0:
        init_test_loss.append(np.average(loss_set))

    if init_test_loss[-1] == min(init_test_loss) and epoch != 0:
        print('Model Saved!')
        torch.save(model.state_dict(), os.path.join(save_path,'trained_model_theta{}_loss10_epoch{}_{}.pth'.format(theta,epoch,np.average(loss_set))))




test(model, test_loader,epoch=0)


num_epochs = 10000
losses = []
loss_avg = []
model.train()
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss1 = 10*criterion(outputs[:,:3], targets[:,:3].to(device))
        loss2 = 100*criterion(outputs[:,3], targets[:,3].to(device))
        # loss1 = criterion(outputs[:,:3], targets[:,:3].to(device))
        # loss2 = 10*criterion(outputs[:,3], targets[:,3].to(device))
        loss = loss1 + loss2
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=100)
        optimizer.step()


        loss_avg.append(loss.item())  # Store the loss value for plotting

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    losses.append(np.average(loss_avg))

    if epoch >= 500 and epoch%500==0:
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




