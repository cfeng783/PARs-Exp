import sys,getopt
sys.path.insert(0, "../")
import numpy as np
from sklearn.ensemble import IsolationForest
from par.utils.data_loader import load_npz_data
from par.utils.metrics import search_best_f1,calc_detection_performance
from par.par_model import PARsModel
from par.signals import ContinuousSignal,CategoricalSignal
from par.autoencoder import Autoencoder
from par.anchor import Anchor
from par.utils.data_util import DataUtil
from anchor import anchor_tabular
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
        opts, args = getopt.getopt(argv, "m:",["admodel="])
    except:
        print(u'python main_rel_eff_study.py --admodel <model>')
        sys.exit(2)
    
    admodel_type = None
    for opt, arg in opts:
        if opt in ['-m','--admodel']:
            if arg.lower() == 'if':
                admodel_type = 'IF'
            elif arg.lower() == 'ae':
                admodel_type = 'AE'
        else:
            print(u'python main_rel_eff_study.py --admodel <model>')
            sys.exit(2)
    
    
    for ds in datasets:
        try:
            train_df,test_df,test_labels,signals = load_npz_data(ds)

            print(f'dataset: {ds}')
            print(f'training size: {len(train_df)}')
            print(f'number of signals: {len(signals)}')
            print('anomaly ratio:',list(test_labels).count(1)/len(test_labels))
            
            du = DataUtil(signals,scaling_method=None)
            extrain_df = du.normalize_and_encode(train_df)
            extest_df = du.normalize_and_encode(test_df)

            minsup = max(10/len(train_df),0.01)
            explainer = PARsModel(signals, wsup=1, wconf=5, minsup=minsup, minconf=0.9, max_predicates_per_rule=5)
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

            # print(train_df.shape, len(feature_names))
            anchor = anchor_tabular.AnchorTabularExplainer(
                class_names=[0,1], feature_names=feature_names, train_data=train_df[feature_names].values)
            
            pos_count = 0
            hit_count = 0
            anchor_count = 0
            f1scores = []
            precisions = []
            recalls = []
            exectimes = []

            anchor_f1, anchor_prec, anchor_rec, anchor_times = [],[],[],[]
            for i in range(len(scores)):
                # print(i,'/',len(isof_scores))
                if scores[i] > th:
                    pos_count += 1
                    start_time = time.time()
                    rules = explainer.find_topk_rules(extest_df.loc[i:i,:], k=5)
                    end_time = time.time()
                    execution_time = end_time - start_time
                    exectimes.append(execution_time)
                    if len(rules)>0:
                        hit_count += 1
                        scores = explainer.score_samples(extest_df, rules)
                        pred_labels = scores>0
                        qm = calc_detection_performance(test_labels,pred_labels)
                        f1scores.append(qm[0])
                        precisions.append(qm[1])
                        recalls.append(qm[2])

                    start_time = time.time()
                    exp = anchor.explain_instance(test_df[feature_names].values[i], admodel.predict, threshold=0.95)
                    end_time = time.time()
                    execution_time = end_time - start_time
                    if len(exp.names())>0:
                        anchor_count += 1
                    anchor_times.append(execution_time)
                    anchor_rule = Anchor(exp)
                    # print(anchor_rule)
                    anchor_labels = anchor_rule.classify_samples(test_df)
                    qm = calc_detection_performance(test_labels,anchor_labels)
                    anchor_f1.append(qm[0])
                    anchor_prec.append(qm[1])
                    anchor_rec.append(qm[2])

            print('PARs vs Anchors:')
            print('F1 score',np.mean(f1scores), np.mean(anchor_f1))
            print('precision',np.mean(precisions), np.mean(anchor_prec))
            print('recall',np.mean(recalls), np.mean(anchor_rec))
            print('exec time',np.mean(exectimes),np.mean(anchor_times))
            print()
        except Exception as e:
            print(e)