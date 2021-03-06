from Data.Metaphors.embeddings import extract_emb
from model import BiLSTMEncoder,MainModel
from helper import evaluate, evaluate_train, get_metaphor_dataset, write_board
import torch.nn as nn
import torch.optim as optim
import torch
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define hyper parameters
    batch_size = 64
    hidden_dim = 300

    loader_dataset = get_metaphor_dataset('Data/Metaphors/VUA/vua_train_embeds.npy','Data/Metaphors/VUA/vua_train_labels.npy',batch_size)
    val_loader = get_metaphor_dataset('Data/Metaphors/VUA/vua_val_embeds.npy','Data/Metaphors/VUA/vua_val_labels.npy',batch_size)

    model = MainModel(embed_dim=1024, hidden_dim = hidden_dim, layers = 1, dropout_lstm = 0, dropout_input=0.5, dropout_FC=0.1, num_classes = 2)
    if torch.cuda.is_available():
        model.to(device=device)
    nll_criterion = nn.NLLLoss()
    # Set up an optimizer for updating the parameters of the rnn_clf
    met_model_optimizer = optim.Adam(model.parameters(), lr=0.01)
    # Number of epochs (passes through the dataset) to train the model for.
    num_epochs = 5

    val_loss = []
    val_f1 = []
    counter = 0

    writer = SummaryWriter()

    for epoch in range(num_epochs):
        print("Starting epoch {}".format(epoch + 1))
        for (data, lengths, labels) in loader_dataset:
            if torch.cuda.is_available():
                data.to(device=device)
                lengths.to(device=device)
                labels.to(device=device)
            predicted = model(data, lengths)
            batch_loss = nll_criterion(predicted.view(-1, 2), labels.view(-1))
            precision, recall, f1, eval_accuracy = evaluate_train(labels, predicted, lengths)
            met_model_optimizer.zero_grad()
            batch_loss.backward()
            met_model_optimizer.step()
            counter += 1
            write_board(writer,'Train', precision, recall, f1, eval_accuracy, batch_loss.item(), counter)
            if counter % 50 == 0:
                avg_eval_loss, precision, recall, f1, eval_accuracy = evaluate(val_loader, model, nll_criterion, device)
                write_board(writer,'Val', precision, recall, f1, eval_accuracy, avg_eval_loss, counter)
                print("Iteration {}. Validation Loss {}. Validation Accuracy {}. Validation Precision {}. Validation Recall {}. Validation F1 {}.".format(counter, avg_eval_loss, eval_accuracy, precision, recall, f1))
    print("First Training done!")

    met_model_optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        print("Starting epoch {}".format(epoch + 1))
        for (data, lengths, labels) in loader_dataset:
            if torch.cuda.is_available():
                data.to(device=device)
                lengths.to(device=device)
                labels.to(device=device)
            predicted = model(data, lengths)
            batch_loss = nll_criterion(predicted.view(-1, 2), labels.view(-1))
            precision, recall, f1, eval_accuracy = evaluate_train(labels, predicted, lengths)
            met_model_optimizer.zero_grad()
            batch_loss.backward()
            met_model_optimizer.step()
            counter += 1
            write_board(writer,'Train', precision, recall, f1, eval_accuracy, batch_loss.item(), counter)
            if counter % 50 == 0:
                avg_eval_loss, precision, recall, f1, eval_accuracy = evaluate(val_loader, model, nll_criterion, device)
                write_board(writer,'Val', precision, recall, f1, eval_accuracy, avg_eval_loss, counter)
                print("Iteration {}. Validation Loss {}. Validation Accuracy {}. Validation Precision {}. Validation Recall {}. Validation F1 {}.".format(counter, avg_eval_loss, eval_accuracy, precision, recall, f1))
    print("First Training done!")

    test_loader = get_metaphor_dataset('Data/Metaphors/VUA/vua_test_embeds.npy','Data/Metaphors/VUA/vua_test_labels.npy',batch_size)
    avg_test_loss, precision, recall, f1, test_accuracy = evaluate(test_loader, model, nll_criterion, device)
    write_board(writer,'Test', precision, recall, f1, test_accuracy, avg_test_loss, counter)
    writer.close()