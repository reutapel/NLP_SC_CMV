import torch as tr
import torch.nn as nn
from torch.autograd import Variable
from model_utils import CustomDataset
from torch.utils import data as dt


class LSTM(nn.Module):
    def __init__(self, embedding_dim_text, nb_lstm_units_text, nb_lstm_layers_text, batch_first_text,
                 embedding_dim_comments, nb_lstm_units_comments, nb_lstm_layers_comments, batch_first_comments,
                 embedding_dim_users, nb_lstm_units_users, nb_lstm_layers_users, batch_first_users):
        # static layers definition- aggregates the parameters for the derivatives
        super(LSTM, self).__init__()

        # define lstm's parameters
        self.embedding_dim_text = embedding_dim_text
        self.nb_lstm_units_text = nb_lstm_units_text
        self.nb_lstm_layers_text = nb_lstm_layers_text
        self.batch_first_text = batch_first_text
        self.embedding_dim_comments = embedding_dim_comments
        self.nb_lstm_units_comments = nb_lstm_units_comments
        self.nb_lstm_layers_comments = nb_lstm_layers_comments
        self.batch_first_comments = batch_first_comments
        self.embedding_dim_users = embedding_dim_users
        self.nb_lstm_units_users = nb_lstm_units_users
        self.nb_lstm_layers_users = nb_lstm_layers_users
        self.batch_first_users = batch_first_users

        # define layers
        self.lstm_text = self.design_lstm(self.embedding_dim_text, self.nb_lstm_units_text, self.nb_lstm_layers_text,
                                          self.batch_first_text)

        self.lstm_comments = self.design_lstm(self.embedding_dim_comments, self.nb_lstm_units_comments,
                                              self.nb_lstm_layers_comments, self.batch_first_comments)

        self.lstm_users = self.design_lstm(self.embedding_dim_users, self.nb_lstm_units_users,
                                           self.nb_lstm_layers_users, self.batch_first_users)

    def design_lstm(self, embedding_dim, nb_lstm_units, nb_lstm_layers, batch_first=True):
        lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=nb_lstm_units,
            num_layers=nb_lstm_layers,
            batch_first=batch_first,)
        return lstm

    def forward(self, x):
        # runs dynamically with every call

        # x = [self.branch_comments_embedded_text_tensor.[index], self.branch_comments_features_tensor[index],
        # self.branch_comments_user_profiles_tensor[index],
        #  self.submission_data_dict[self.branch_submission_dict[index]] =
        # [submission text, submission features, submitter profile features] ]

        branch_comments_embedded_text = x[0]
        branch_comments_features_tensor = x[1]
        branch_comments_user_profiles_tensor = x[2]
        submission_text = x[3][0]
        submission_features = x[3][1]
        submitter_profile_features = x[3][2]
        # TODO: add smart concatenation of 3 vectors with 3 lstm
        output_text = self.lstm_text(branch_comments_embedded_text)
        output_comments = self.lstm_comments(branch_comments_features_tensor)
        output_users = self.lstm_users(branch_comments_user_profiles_tensor)

        return

# TODO: ensure zero weights initializing between batches
# TODO: all pack padded issue
# TODO: build logic concatenation in forward

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
model=LSTM()
# on the model parameters will be performed the update
optimizer = tr.optim.SGD(model.parameters(), lr=learning_rate)
batch_size = 128
num_epochs = 100

train_dataset = CustomDataset(branch_comments_embedded_text_df, branch_comments_features_df,
                              branch_comments_user_profiles_df, branch_submission_dict, submission_data_dict,
                              branch_deltas_data_dict)

test_dataset = CustomDataset(branch_comments_embedded_text_df, branch_comments_features_df,
                              branch_comments_user_profiles_df, branch_submission_dict, submission_data_dict,
                              branch_deltas_data_dict)


train_loader = dt.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = dt.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)



# training
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

# TODO: add batch normalization
# TODO: F.mse_loss(size_average, reduce) : parameters that affect if we get average values per batch : sum or average