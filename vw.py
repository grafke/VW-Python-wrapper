from __future__ import division
import csv
import os
import subprocess
from time import time
import math

__author__ = 'grf'

class VW():
    '''
    Generic VW object
    '''

    def __init__(self, args=None,
                 arg_string=None):
        """
        Default constructor
        :param args:
        :param arg_string:
        """
        self.arg_list = arg_string
        self.parsed_args = args
        self._features = set()
        self._feature_weights = set()
        self._delimiter = ':'
        self.training_time = 0.
        self.audit_time = 0.
        self.sparsity = 0

    def __get_audit_log(self, audit_log):
        """
        Returns the list of all feature-names
        :param audit_log:
        """
        vw_audit_args = self.parsed_args.vw + ' --quiet -t --audit -i ' + self.parsed_args.final_regressor + ' -d ' + self.parsed_args.data
        vw_audit_args = vw_audit_args.split()
        f = open(audit_log,'w')
        proc = subprocess.Popen(vw_audit_args, shell=False, stdout=f).communicate()
        os.fsync(f)
        f.close()
        with open(audit_log, 'r') as f:
            while True:
                f.readline()
                line = f.readline()
                if not line: break
                self.__audit_one_example(line)

    @staticmethod
    def __save_to_file(filename, result):
        """
        Saves a collection to a file
        :param filename:
        :param result:
        """
        audit_log_file = open(filename, 'w')
        for line in result:
            audit_log_file.write(line)
        audit_log_file.close()

    def __audit_one_example(self, result):
        """
        Audited feature format:   Namespace^featureName:142703:1:0.0435613
        :param result:
        """
        f = [x.split(self._delimiter) for x in result.strip().split('\t')]
        for x in f:
            self.__add_feature(x)

    def __add_feature(self, example):
        """
        Builds a feature set
        :param example:
        """
        if float(example[-1]) != 0. and example[0] not in self._features:
            self._features.add(example[0])
            example[0] = example[0].replace('^', '\t')
            self._feature_weights.add('\t'.join(example) + '\n')

    @property
    def __collect_predictions(self):
        """
        Collects predictions. Supports logistic loss function only
        :return:
        """
        inf = open(self.parsed_args.predictions, 'r')
        reader = csv.reader(inf)
        predictions = [self.__sigmoid(row) for row in reader]
        inf.close()
        return predictions

    @staticmethod
    def __sigmoid(param):
        """
        Sigmoid function
        :param param:
        :return: sigmoid
        """
        x = float(param[0])
        return 1 / (1 + math.exp(-x))

    @property
    def __collect_real_class_values(self):
        """
        Collects class values from the test file
        :return:
        """
        inf = open(self.parsed_args.testonly, 'r')
        reader = csv.reader(inf, delimiter=' ')
        real_class_values = [int(x[0]) for x in reader]
        inf.close()
        return real_class_values

    @staticmethod
    def __get_roc_auc(real_classes, predictions):
        from sklearn.metrics import roc_curve, auc
        """
        :param real_classes:
        :param predictions:
        :return: true positive rate, false positive rate, thresholds, AUC
        """
        fpr, tpr, thresholds = roc_curve(real_classes, predictions, pos_label=1.0)
        roc_auc = auc(fpr, tpr)
        return fpr, tpr, thresholds, roc_auc

    @staticmethod
    def __get_ne(real_classes, predictions):
        from sklearn.metrics import log_loss
        from numpy import log
        rctr = real_classes.count(1) / len(real_classes)
        entropy = -(rctr * log(rctr) + (1 - rctr) * log(1 - rctr))
        return log_loss(real_classes, predictions) / entropy

    def learn(self):
        """
        Runs the training stage to compute per feature weights
        """
        start = time()
        subprocess.Popen(self.arg_list).communicate()
        self.training_time = time() - start

    @property
    def test(self):
        """
        Evaluates the learned model and calculates AUC
        :rtype : AUC
        """
        params = [self.parsed_args.vw, '-t', self.parsed_args.testonly, '-k', '-c', '--cache_file',
                  self.parsed_args.cache_file, '-i', self.parsed_args.initial_regressor,
                  '-p', self.parsed_args.predictions, '--hash', self.parsed_args.hash,
                  '--bit_precision', str(self.parsed_args.bit_precision)]
        subprocess.Popen(params).communicate()

        #
        predictions = self.__collect_predictions
        ectr=sum(predictions)/len(predictions)
        real_classes = self.__collect_real_class_values
        rctr=real_classes.count(1)/len(real_classes)
        fpr, tpr, thresholds, roc_auc = self.__get_roc_auc(real_classes, predictions)
        return roc_auc, ectr, rctr

    def summarize_features(self, audit_log='', summary_file='', save_summary=False):
        """
        Outputs what we know about all features. Only one loop for non multi-class. Multi-class is not supported yet.
        :param audit_log:
        :param summary_file:
        :param save_summary:
        """
        start = time()
        self.__get_audit_log(audit_log)
        self.audit_time = time() - start
        self.sparsity = len(self._features)
        if save_summary:
            self.__save_to_file(summary_file, self._feature_weights)