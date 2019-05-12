from Data.Metaphors.embeddings import extract_emb
from model import BiLSTMEncoder,MainModel, ModelHyper
from helper import evaluate, evaluate_train, get_metaphor_dataset, write_board, get_document_dataset, evaluate_train_hyper,evaluate_hyper
import torch.nn as nn
import torch.optim as optim
import torch
from tensorboardX import SummaryWriter
import time
import argparse


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--dropout_lstm', type=int, default=0, help='Number of hidden units in the model')
    parser.add_argument('--dropout_input', type=int, default=0.5, help='Number of hidden units in the model')
    parser.add_argument('--dropout_FC', type=int, default=0.1, help='Number of hidden units in the model')
    parser.add_argument('--dropout_lstm_hyper', type=int, default=0.3, help='Number of hidden units in the model')
    parser.add_argument('--dropout_input_hyper', type=int, default=0, help='Number of hidden units in the model')
    parser.add_argument('--dropout_attention', type=int, default=0, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training steps')
    parser.add_argument('--hidden_dim', type=float, default=300)
    parser.add_argument('--device', type=str, default="cuda:0", help="Training device 'cpu' or 'cuda:0'")

    config = parser.parse_args()

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define hyper parameters
    batch_size = config.batch_size
    hidden_dim = config.hidden_dim
    
    loader_dataset_hyp, loader_val_hyp = get_document_dataset('Data/Hyperpartisan/hyp_embeds.npy','Data/Hyperpartisan/hyp_labels.npy',batch_size)

    model = ModelHyper(embed_dim=1024, 
                        hidden_dim = hidden_dim, 
                        layers = 1, 
                        dropout_lstm = config.dropout_lstm, 
                        dropout_input=config.dropout_input, 
                        dropout_FC=config.dropout_FC,
                        dropout_lstm_hyper = config.dropout_lstm_hyper,
                        dropout_input_hyper = config.dropout_input_hyper,
                        dropout_attention = config.dropout_attention,
                        num_classes = 2)
    if torch.cuda.is_available():
        model.to(device=device)
    nll_criterion = nn.NLLLoss()
    # Set up an optimizer for updating the parameters of the rnn_clf
    met_model_optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # Number of epochs (passes through the dataset) to train the model for.
    num_epochs = config.epochs

    val_loss = []
    val_f1 = []
    counter = 0

    writer = SummaryWriter()

    for epoch in range(num_epochs):
        print("Starting epoch {}".format(epoch + 1))
        for (data, doc_len, labels, sen_len) in loader_dataset_hyp:
            if torch.cuda.is_available():
                doc_len = doc_len.to(device=device)
                labels = labels.to(device=device)
            start = time.time()
            predicted = model(data, sen_len, doc_len)
            end = time.time()
            print(end - start, ' Forward pass')
            batch_loss = nll_criterion(predicted.view(-1, 2), labels.view(-1))
            precision, recall, f1, eval_accuracy = evaluate_train_hyper(labels, predicted)
            end = time.time()
            print(end - start, ' Evaluate and compute loss')
            met_model_optimizer.zero_grad()
            batch_loss.backward()
            end = time.time()
            print(end - start, ' Backward pass')
            met_model_optimizer.step()
            end = time.time()
            print(end - start, ' optimize step')
            counter += 1
            write_board(writer,'Hyper/Train', precision, recall, f1, eval_accuracy, batch_loss.item(), counter)
            if counter % 2 == 0:
                avg_eval_loss, precision, recall, f1, eval_accuracy = evaluate_hyper(loader_val_hyp, model, nll_criterion, device)
                write_board(writer,'Hyper/Val', precision, recall, f1, eval_accuracy, avg_eval_loss, counter)
                print("Iteration {}. Validation Loss {}. Validation Accuracy {}. Validation Precision {}. Validation Recall {}. Validation F1 {}.".format(counter, avg_eval_loss, eval_accuracy, precision, recall, f1))
    print("First Training done!")

    '''    test_loader = get_metaphor_dataset('Data/Metaphors/VUA/vua_test_embeds.npy','Data/Metaphors/VUA/vua_test_labels.npy',batch_size)
    avg_test_loss, precision, recall, f1, test_accuracy = evaluate(test_loader, model, nll_criterion, device)
    write_board(writer,'Test', precision, recall, f1, test_accuracy, avg_test_loss, counter)'''
    writer.close()
    checkpoint = {'hyperparameters': config,
              'state_dict': model.state_dict()}
    torch.save(checkpoint, 'hyper_model.pt')
    #torch.save(model.state_dict(), 'Model/hyper.pt')