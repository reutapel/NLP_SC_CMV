import torch as tr
import torch.nn as nn
from torch.autograd import Variable
from model_utils import CustomDataset
from torch.utils import data as dt
from DeltaModel import DeltaModel


learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
model = DeltaModel()
# on the model parameters will be performed the update
optimizer = tr.optim.SGD(model.parameters(), lr=learning_rate)
batch_size = 128
num_epochs = 100


# create data sets
train_dataset = CustomDataset(branch_comments_embedded_text_df_train, branch_comments_features_df_train,
                              branch_comments_user_profiles_df_train, branch_submission_dict_train, submission_data_dict_train,
                              branch_deltas_data_dict_train)

test_dataset = CustomDataset(branch_comments_embedded_text_df_test, branch_comments_features_df_test,
                              branch_comments_user_profiles_df_test, branch_submission_dict_test, submission_data_dict_test,
                              branch_deltas_data_dict_test)

# create data loaders for model training
train_loader = dt.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = dt.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


# training

device = tr.device("cuda" if tr.cuda.is_available() else "cpu")

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # -1 so last batch will fit the size
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)
        # forward + backward + optimize
        # initialize gradient so only current batch will be summed and then backward
        optimizer.zero_grad()
        # forward
        outputs = model(images)
        # TODO: understand impact of packed padded to loss, like function loss in model.py
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if (i+1) % 100 ==0:
            print('Epoch: [%d%d], Step: [%d%d], Loss: %.4f' % (epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))


# testing

correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    outputs = model(images)
    _, predicted = tr.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum()
    break
print('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# save the model
tr.save(model.state_dict(), 'model.pkl')

# TODO: add batch normalization, dropout & activation functions
# TODO: add metric learning for a richer representation of the label
# TODO: F.mse_loss(size_average, reduce) : parameters that affect if we get average values per batch : sum or average