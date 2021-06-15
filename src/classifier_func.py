
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import statistics
import tqdm
import sklearn
import random
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from utils import *

def evaluate(true_y, pred_y, pred_prob):
    ## Accuracy
    accuracy = metrics.accuracy_score(true_y, pred_y)
    accuracy = round(accuracy, 3)

    ## AUC
    fpr, tpr, thresholds = metrics.roc_curve(true_y, pred_prob)
    auc_score = metrics.auc(fpr, tpr)
    auc_score = round(auc_score, 3)

    # F1 micro
    # f1_micro = metrics.f1_score(true_y, pred_y, average="micro")
    # f1_macro = metrics.f1_score(true_y, pred_y, average="macro")

    # return accuracy, auc_score, f1_micro, f1_macro
    return accuracy, auc_score

def random_forest(max_depths, train_x, train_labels_arr, valid_x, valid_labels_arr, test_x, test_labels_arr):

    ## Random Forest Classifier
    print("*** Random Forest Classifier: ")
    rf_auc_dict = {"train": [], "valid": [], "test": []}
    for max_depth in max_depths:
        print("Max depth: {}".format(max_depth))
        repeat_performances = {"train":{"accuracy": [], "auc":[]},\
                                "valid": {"accuracy": [], "auc":[]},\
                                "test": {"accuracy": [], "auc":[]}}
        for repeat in tqdm(range(10)):
            clf = RandomForestClassifier(n_estimators=1000, max_depth=max_depth, random_state=repeat)
            clf.fit(train_x, train_labels_arr)
            train_pred = clf.predict(train_x)
            train_pred_prob = clf.predict_proba(train_x)[:,np.where(clf.classes_==1)[0]]
            train_accuracy, train_auc = evaluate(train_labels_arr, train_pred, train_pred_prob)

            repeat_performances["train"]["accuracy"].append(train_accuracy)
            repeat_performances["train"]["auc"].append(train_auc)

            valid_pred = clf.predict(valid_x)
            valid_pred_prob = clf.predict_proba(valid_x)[:,np.where(clf.classes_==1)[0]]
            valid_accuracy, valid_auc = evaluate(valid_labels_arr, valid_pred, valid_pred_prob)

            repeat_performances["valid"]["accuracy"].append(valid_accuracy)
            repeat_performances["valid"]["auc"].append(valid_auc)

            test_pred = clf.predict(test_x)
            test_pred_prob = clf.predict_proba(test_x)[:,np.where(clf.classes_==1)[0]]
            test_accuracy, test_auc = evaluate(test_labels_arr, test_pred, test_pred_prob)
            
            repeat_performances["test"]["accuracy"].append(test_accuracy)
            repeat_performances["test"]["auc"].append(test_auc)

        # np.savez("temp_data/baseline2_rf3_test_auc.npz", auc=np.array(repeat_performances["test"]["auc"]))
        avg_train_accuracy = statistics.mean(repeat_performances["train"]["accuracy"])
        avg_train_auc = statistics.mean(repeat_performances["train"]["auc"])
        avg_valid_accuracy = statistics.mean(repeat_performances["valid"]["accuracy"])
        avg_valid_auc = statistics.mean(repeat_performances["valid"]["auc"])
        avg_test_accuracy = statistics.mean(repeat_performances["test"]["accuracy"])
        avg_test_auc = statistics.mean(repeat_performances["test"]["auc"])

        rf_auc_dict["train"].append(avg_train_auc)
        rf_auc_dict["valid"].append(avg_valid_auc)
        rf_auc_dict["test"].append(avg_test_auc)

        print("Max depth={}: \n \
        Train accuracy: {:.3f}, auc: {:.3f}\n \
        Valid accuracy: {:.3f}, auc: {:.3f}\n \
        Test accuracy: {:.3f}, auc: {:.3f}\n".format(max_depth, avg_train_accuracy, avg_train_auc, avg_valid_accuracy, avg_valid_auc, avg_test_accuracy, avg_test_auc))

    # select the hyper-parameter with the highest valid accuracy
    max_idx = rf_auc_dict["valid"].index(max(rf_auc_dict["valid"]))

    print("Best model performances: max_depth={}".format(max_depths[max_idx]))

    return rf_auc_dict 

def svm(cs, train_x, train_labels_arr, valid_x, valid_labels_arr, test_x, test_labels_arr):
    ## Support vector machines
    print("*** SVM: ")
    svm_auc_dict = {"train": [], "valid": [], "test": []}
    for c in cs:
        print("C: {}".format(c))
        repeat_performances = {"train":{"accuracy": [], "auc":[]},\
                                "valid": {"accuracy": [], "auc":[]},\
                                "test": {"accuracy": [], "auc":[]}}
        for repeat in tqdm(range(10)):
            clf = SVC(C=c, kernel="linear", gamma="scale", probability=True, random_state=repeat)
            clf.fit(train_x, train_labels_arr)
            train_pred = clf.predict(train_x)
            train_pred_prob = clf.predict_proba(train_x)[:,np.where(clf.classes_==1)[0]]
            train_accuracy, train_auc = evaluate(train_labels_arr, train_pred, train_pred_prob)
            repeat_performances["train"]["accuracy"].append(train_accuracy)
            repeat_performances["train"]["auc"].append(train_auc)

            valid_pred = clf.predict(valid_x)
            valid_pred_prob = clf.predict_proba(valid_x)[:,np.where(clf.classes_==1)[0]]
            valid_accuracy, valid_auc = evaluate(valid_labels_arr, valid_pred, valid_pred_prob)
            repeat_performances["valid"]["accuracy"].append(valid_accuracy)
            repeat_performances["valid"]["auc"].append(valid_auc)

            test_pred = clf.predict(test_x)
            test_pred_prob = clf.predict_proba(test_x)[:,np.where(clf.classes_==1)[0]]
            test_accuracy, test_auc = evaluate(test_labels_arr, test_pred, test_pred_prob)
            repeat_performances["test"]["accuracy"].append(test_accuracy)
            repeat_performances["test"]["auc"].append(test_auc)


        avg_train_accuracy = statistics.mean(repeat_performances["train"]["accuracy"])
        avg_train_auc = statistics.mean(repeat_performances["train"]["auc"])
        avg_valid_accuracy = statistics.mean(repeat_performances["valid"]["accuracy"])
        avg_valid_auc = statistics.mean(repeat_performances["valid"]["auc"])
        avg_test_accuracy = statistics.mean(repeat_performances["test"]["accuracy"])
        avg_test_auc = statistics.mean(repeat_performances["test"]["auc"])

        svm_auc_dict["train"].append(avg_train_auc)
        svm_auc_dict["valid"].append(avg_valid_auc)
        svm_auc_dict["test"].append(avg_test_auc)

        print("C={}: \n \
        Train accuracy: {:.3f}, auc: {:.3f}\n \
        Valid accuracy: {:.3f}, auc: {:.3f}\n \
        Test accuracy: {:.3f}, auc: {:.3f}\n".format(c, avg_train_accuracy, avg_train_auc, avg_valid_accuracy, avg_valid_auc, avg_test_accuracy, avg_test_auc))

    # select the hyper-parameter with the highest valid accuracy
    max_idx = svm_auc_dict["valid"].index(max(svm_auc_dict["valid"]))

    print("Best model performances: C={}".format(cs[max_idx]))

    return svm_auc_dict

class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

def mlp(hidden_nodes, normalized_train_x, train_labels, normalized_valid_x, valid_labels, normalized_test_x, test_labels):
    loss = nn.BCEWithLogitsLoss()
    epochs = 300


    patience = 7

    # dictionaries that store average auc and accuracy for each hidden node
    mlp_loss_dict = {"train":[], "valid":[], "test":[]}
    mlp_auc_dict = {"train":[], "valid":[], "test":[]}

    for hidden_node in hidden_nodes:
        print("Hidden node: {}".format(hidden_node))
        # dictionaries that store repeated performances
        repeat_performances = {"train":{"loss":[], "accuracy": [], "auc":[]},\
                                "valid": {"loss":[], "accuracy": [], "auc":[]},\
                                "test": {"loss":[], "accuracy": [], "auc":[]}}
        for repeat in tqdm(range(10)):
            torch.manual_seed(repeat)
            model = MLPClassifier(input_dim=normalized_train_x.shape[1], hidden_dim=hidden_node, output_dim=1)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            # Early stopping
            stopper = EarlyStopping(patience=30)

            train_loss_arr = np.zeros((epochs))
            valid_loss_arr = np.zeros((epochs))
            test_loss_arr = np.zeros((epochs))
            train_auc_arr = np.zeros((epochs))
            valid_auc_arr = np.zeros((epochs))
            test_auc_arr = np.zeros((epochs))
            for epoch in range(epochs):
                model.train()

                optimizer.zero_grad()

                train_y_hat = model(normalized_train_x)
                train_loss = loss(train_y_hat, train_labels.view(train_labels.shape[0],1))
                train_loss_arr[epoch] = train_loss
                # get the train predictions
                train_prob = torch.sigmoid(train_y_hat)
                train_pred = torch.round(train_prob)
                train_auc = round(roc_auc_score(train_labels.detach().numpy(), train_prob.detach().numpy()), 3)
                train_auc_arr[epoch] = train_auc

                train_loss.backward()
                optimizer.step()

                # predict on validation and test sets
                with torch.no_grad():
                    model.eval()

                    # # predict on the valid set
                    valid_y_hat = model(normalized_valid_x)
                    valid_loss = loss(valid_y_hat, valid_labels.view(valid_labels.shape[0],1))
                    valid_loss_arr[epoch] = valid_loss
                    # get the valid predictions
                    valid_prob = torch.sigmoid(valid_y_hat)
                    valid_pred = torch.round(valid_prob)
                    valid_auc = round(roc_auc_score(valid_labels.detach().numpy(), valid_prob.detach().numpy()), 3)
                    valid_auc_arr[epoch] = valid_auc

                    # predict on the test set
                    test_y_hat = model(normalized_test_x)
                    test_loss = loss(test_y_hat, test_labels.view(test_labels.shape[0],1))
                    test_loss_arr[epoch] = test_loss
                    # get the test predictions
                    test_prob = torch.sigmoid(test_y_hat)
                    test_pred = torch.round(test_prob)
                    test_auc = round(roc_auc_score(test_labels.detach().numpy(), test_prob.detach().numpy()), 3)
                    test_auc_arr[epoch] = test_auc

                if stopper.step(valid_loss, model):
                    best_score = stopper.best_score
                    best_epoch = np.where(valid_loss_arr == best_score)[0][0]

                    train_loss = train_loss_arr[best_epoch]
                    train_auc = train_auc_arr[best_epoch]
                    valid_loss = valid_loss_arr[best_epoch]
                    valid_auc = valid_auc_arr[best_epoch]
                    test_loss = test_loss_arr[best_epoch]
                    test_auc = test_auc_arr[best_epoch]
                    break

            repeat_performances["train"]["loss"].append(float(train_loss))
            repeat_performances["train"]["auc"].append(train_auc)
            repeat_performances["valid"]["loss"].append(float(valid_loss))
            repeat_performances["valid"]["auc"].append(valid_auc)
            repeat_performances["test"]["loss"].append(float(test_loss))
            repeat_performances["test"]["auc"].append(test_auc)

            # save 100 repeated performance (test)
        # np.savez("temp_data/baseline1_mlp40_test_auc.npz", baseline1=np.array(repeat_performances["test"]["auc"]))

        avg_train_loss = statistics.mean(repeat_performances["train"]["loss"])
        avg_train_auc = statistics.mean(repeat_performances["train"]["auc"])
        avg_valid_loss = statistics.mean(repeat_performances["valid"]["loss"])
        avg_valid_auc = statistics.mean(repeat_performances["valid"]["auc"])
        avg_test_loss = statistics.mean(repeat_performances["test"]["loss"])
        avg_test_auc = statistics.mean(repeat_performances["test"]["auc"])

        mlp_loss_dict["train"].append(avg_train_loss)
        mlp_loss_dict["valid"].append(avg_valid_loss)
        mlp_loss_dict["test"].append(avg_test_loss)
        mlp_auc_dict["train"].append(avg_train_auc)
        mlp_auc_dict["valid"].append(avg_valid_auc)
        mlp_auc_dict["test"].append(avg_test_auc)

        print("Hidden node={}:\n Train Loss: {:.3f}, auc: {:.3f}\n \
        Valid Loss: {:.3f}, auc: {:.3f}\n\
        Test Loss: {:.3f}, auc: {:.3f}\n".format(hidden_node, avg_train_loss, avg_train_auc, avg_valid_loss, avg_valid_auc, avg_test_loss, avg_test_auc))

    # find the best hidden node
    max_idx = mlp_auc_dict["valid"].index(max(mlp_auc_dict["valid"]))
    print("Best model performances: hidden node={}".format(hidden_nodes[max_idx]))
    print("Train loss: {:.3f}, auc: {:.3f}\n Valid loss: {:.3f}, auc: {:.3f}\n Test loss: {:.3f}, auc: {:.3f}".format(mlp_loss_dict["train"][max_idx], mlp_auc_dict["train"][max_idx], mlp_loss_dict["valid"][max_idx], mlp_auc_dict["valid"][max_idx], mlp_loss_dict["test"][max_idx], mlp_auc_dict["test"][max_idx]))

    return mlp_auc_dict


