from Data.Metaphors.embeddings import extract_emb
from model import BiLSTMEncoder,MainModel, ModelHyper
from helper import evaluate, evaluate_train, get_metaphor_dataset, write_board, get_document_dataset, evaluate_train_hyper,evaluate_hyper
import torch.nn as nn
import torch.optim as optim
import torch
from tensorboardX import SummaryWriter
import time

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Define hyper parameters
    batch_size = 16
    hidden_dim = 300
    
    loader_dataset_hyp, loader_val_hyp = get_document_dataset('Data/Hyperpartisan/hyp_embeds.npy','Data/Hyperpartisan/hyp_labels.npy',batch_size)

    model = ModelHyper(embed_dim=1024, 
                        hidden_dim = hidden_dim, 
                        layers = 1, 
                        dropout_lstm = 0, 
                        dropout_input=0.5, 
                        dropout_FC=0.1,
                        dropout_lstm_hyper = 0.3,
                        dropout_input_hyper = 0.6,
                        dropout_attention = 0.4,
                        num_classes = 2)
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
        for (data, doc_len, labels, sen_len) in loader_dataset_hyp:
            if torch.cuda.is_available():
                data.to(device=device)
                lengths.to(device=device)
                labels.to(device=device)
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
    torch.save(model.state_dict(), 'Model/hyper.pt')