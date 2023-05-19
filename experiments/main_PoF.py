'''
Created on May 18, 2023

@author: z003w5we
'''
import sys,getopt
sys.path.insert(0, "../")
import numpy as np
from sklearn.ensemble import IsolationForest
from par.utils.data_loader import load_npz_data
from par.utils.metrics import search_best_f1
from par.par_model import PARsModel,PredicateMode
from par.signals import ContinuousSignal,CategoricalSignal,BaseSignal
from par.autoencoder import Autoencoder
from par.utils.data_util import DataUtil
import time
import warnings

datasets = ['4_breastw','6_cardio','7_Cardiotocography','12_fault','18_Ionosphere','21_Lymphography',
            '22_magic','29_Pima','30_satellite', '31_satimage-2',
            '32_shuttle',  '33_skin', '37_Stamps', '38_thyroid','42_WBC', '43_WDBC','45_wine']


class AnomalyDetector:
    def __init__(self, model, th, mtype):
        '''
        Constructor
        '''
        self.model = model
        self.th = th
        self.type = mtype
        
    def predict(self, x):
        scores = self.model.score_samples(x)
        if self.type == 'IF':
            scores = -1*scores
        y = []
        for score in scores:
            if score > self.th:
                y.append(1)
            else:
                y.append(0)
        return np.array(y)
        
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    argv = sys.argv[1:]
    
    try:
        opts, args = getopt.getopt(argv, "m:p:",["admodel=","pm="])
    except:
        print(u'python main_PoF.py --admodel <model> --pm <predicate mode>')
        sys.exit(2)
    
    admodel_type = None
    pm = PredicateMode.Dependency
    for opt, arg in opts:
        if opt in ['-m','--admodel']:
            if arg.lower() == 'if':
                admodel_type = 'IF'
            elif arg.lower() == 'ae':
                admodel_type = 'AE'
        elif opt in ['-p','--pm']:
            if arg== '0':
                pm = PredicateMode.Dependency
            elif arg == '1':
                pm = PredicateMode.KMeansBins
            elif arg == '2':
                pm = PredicateMode.UniformBins
        else:
            print(u'python main_PoF.py --admodel <model> --pm <predicate mode>')
            sys.exit(2)
    
    
    for ds in datasets:
        try:
            train_df,test_df,test_labels,signals = load_npz_data(ds)

            print(f'dataset: {ds}')
            print(f'number of signals: {len(signals)}')
            
            du = DataUtil(signals,scaling_method=None)
            extrain_df = du.normalize_and_encode(train_df)
            extest_df = du.normalize_and_encode(test_df)

            minsup = max(10/len(train_df),0.01)
            explainer = PARsModel(signals, wsup=1, wconf=5, minsup=minsup, minconf=0.9, max_predicates_per_rule=5, mode=pm)
            explainer.train(extrain_df, max_perdicts4rule_mining = 100, max_times4rule_mining = 5, verbose=False)
            
            if admodel_type == 'IF':
                feature_names = [signal.name for signal in signals]
                model = IsolationForest()
                model.fit(train_df[feature_names].values)
                scores = model.score_samples(test_df[feature_names].values)
                scores = -1*scores
            elif admodel_type == 'AE':
                du = DataUtil(signals,scaling_method='min_max')
                train_df = du.normalize_and_encode(train_df)
                test_df = du.normalize_and_encode(test_df)
                
                feature_names = []
                for signal in signals:
                    if isinstance(signal, ContinuousSignal):
                        feature_names.append(signal.name)
                    if isinstance(signal, CategoricalSignal):
                        feature_names.extend(signal.get_onehot_feature_names())
                model = Autoencoder(signals)
                model.train(train_df[feature_names].values, hidden_dim=len(signals)//3, num_hidden_layers=2, batch_size=64, epochs=10)
                scores = model.score_samples(test_df[feature_names].values)
        
            qm, th = search_best_f1(scores, test_labels, min(scores), end=max(scores), step_num=1000, verbose=False)
            admodel = AnomalyDetector(model, th,admodel_type)   

            tt_count = 0.001
            tt_hit = 0
            tp_count = 0.001
            fp_count = 0.001
            tp_hit = 0
            fp_hit = 0
            tt_scores = []
            for i in range(len(scores)):
                # print(i,'/',len(isof_scores))
                if scores[i] > th:
                    tt_count += 1
                    start_time = time.time()
                    rules = explainer.find_topk_rules(extest_df.loc[i:i, :], k=1)
                    end_time = time.time()
                    execution_time = end_time - start_time


                    if len(rules) > 0:
                        if isinstance(rules[0], BaseSignal):
                            rconf = 1
                            rsup = 1
                        else:
                            rconf = rules[0].conf
                            rsup = rules[0].support

                        tt_hit += 1
                        
                        par_score = 5*(rconf-0.9)/(1-0.9)+ (rsup-minsup)/(1-minsup)
                        
                        tt_scores.append(par_score)

                    if test_labels[i] == 1: ##tp
                        tp_count += 1
                        if len(rules) > 0:
                            tp_hit += 1
                    else:
                        fp_count += 1
                        if len(rules) > 0:
                            fp_hit += 1

            print('PoF@TP', tp_hit/tp_count)
            print('PoF@FP', fp_hit/fp_count)
            print('Top1 PAR Score', np.mean(tt_scores))
            print()
        except Exception as e:
            print(e)