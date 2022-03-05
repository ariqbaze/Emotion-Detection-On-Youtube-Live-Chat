from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from PyQt5.QtWidgets import QTextEdit, QMainWindow
from PyQt5 import uic


class ConfusionMatrix(QMainWindow):
    def __init__(self, y_test, y_pred, index):
        super(ConfusionMatrix, self).__init__()
        # Load UI
        uic.loadUi("ui/LogMatrix.ui", self)

        self.text = self.findChild(QTextEdit, "textEdit")

        text = ""

        text = text + ('Accuracy: {:.2f}\n'.format(accuracy_score(y_test, y_pred)))

        text = text +('Micro Precision: {:.2f} \n'.format(precision_score(y_test, y_pred, average='micro')))
        text = text +('Micro Recall: {:.2f} \n'.format(recall_score(y_test, y_pred, average='micro')))
        text = text +('Micro F1-score: {:.2f} \n'.format(f1_score(y_test, y_pred, average='micro')))

        text = text +('Macro Precision: {:.2f} \n'.format(precision_score(y_test, y_pred, average='macro')))
        text = text +('Macro Recall: {:.2f} \n'.format(recall_score(y_test, y_pred, average='macro')))
        text = text +('Macro F1-score: {:.2f} \n'.format(f1_score(y_test, y_pred, average='macro')))

        text = text +('Weighted Precision: {:.2f} \n'.format(precision_score(y_test, y_pred, average='weighted')))
        text = text +('Weighted Recall: {:.2f} \n'.format(recall_score(y_test, y_pred, average='weighted')))
        text = text +('Weighted F1-score: {:.2f} \n'.format(f1_score(y_test, y_pred, average='weighted')))

        text = text +('\nClassification Report\n')
        text = text +(classification_report(y_test, y_pred))
        text = text +(index)

        self.text.setText(text)
        self.show()
