import sys,getopt
sys.path.insert(0, "../")
import warnings
from sklearn.ensemble import IsolationForest
from par.utils.data_loader import load_perturb_npz_data
from par.par_model import PARsModel
from par.signals import ContinuousSignal,CategoricalSignal,BaseSignal
from par.utils.metrics import hitRates
from par.autoencoder import Autoencoder
from par.utils.data_util import DataUtil
import numpy as np
import shap,lime



datasets = ['4_breastw','6_cardio','7_Cardiotocography','12_fault','18_Ionosphere','21_Lymphography',
            '22_magic','29_Pima','30_satellite', '31_satimage-2',
            '32_shuttle',  '33_skin', '37_Stamps', '38_thyroid','42_WBC', '43_WDBC','45_wine']

def identify_abfeats_PAR(rules):
    sfs = []
    bfs = []
    for rule in rules:
        if isinstance(rule, BaseSignal):
            feat = rule.name
            if feat not in sfs:
                sfs.append(feat)
        else:
            for feat in rule.extract_feats()[1:]:
                if feat not in bfs:
                    bfs.append(feat)
            feat = rule.extract_feats()[0]
            if feat not in sfs:
                sfs.append(feat)
    for feat in bfs:
        if feat not in sfs:
            sfs.append(feat)
    return sfs



if __name__ == '__main__':
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
        train_df,test_df,pdims,signals = load_perturb_npz_data(ds)
        
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
            train_scores = model.score_samples(train_df[feature_names].values)
            train_scores = train_scores*-1
            
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
            train_scores = model.score_samples(train_df[feature_names].values)
        
        pot_labels = scores > np.quantile(train_scores,0.95)
        x_train = train_df[feature_names].values
        x_test = test_df[feature_names].values
        
        shap_explainer = shap.KernelExplainer(model.score_samples, shap.kmeans(x_train, 20))
        
        lime_exp = lime.lime_tabular.LimeTabularExplainer(x_train,mode='regression', feature_names=feature_names, class_names=[-1], discretize_continuous=True)
        
        hit_rates = []
        shap_rates = []
        lime_rates = []
        for i in range(len(pot_labels)):
            if pot_labels[i] == 1:
                gts = ['col'+str(pd) for pd in pdims[i]]
                rules = explainer.find_topk_rules(extest_df.loc[i:i,:], k=5)
                PAR_pts = identify_abfeats_PAR(rules)
                hit_rates.append(hitRates(gts, PAR_pts))
                
                k = max(len(signals),5)
                
                shap_values = shap_explainer.shap_values(x_test[i])
                shap_values = shap_values*-1
                shap_pts = np.array(shap_values).argsort()[-k:][::-1]
                shap_rates.append(hitRates(pdims[i], shap_pts))
                
                exp = lime_exp.explain_instance(x_test[i], model.score_samples, num_features=k)
                lime_pts = []
                for tup in exp.as_map()[0]:
                    lime_pts.append(tup[0])
                lime_rates.append(hitRates(pdims[i], lime_pts))
        print(f'dataset: {ds}')
        print('PAR rates:', np.mean(np.array(hit_rates),axis=0))
        print('SHAP rates:', np.mean(np.array(shap_rates),axis=0))
        print('LIME rates:', np.mean(np.array(lime_rates),axis=0))
        print()
            