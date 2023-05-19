from .par_base import BaseModel
import pickle
from .rule_miner import mine_rules
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from .signals import ContinuousSignal,BaseSignal
import numpy as np
from . import helper
from sklearn.preprocessing import KBinsDiscretizer
# from .anomaly_explanation import AnomalyExplanation
from enum import Enum
class PredicateMode(Enum):
    UniformBins = 201
    KMeansBins = 202
    Dependency = 203

class PARsModel(BaseModel):
    '''
    Predicate-based Association Rules
    
    Parameters
    ----------
    signals : list
        the list of signals the model is dealing with
    '''
    def __init__(self, signals, wsup=1, wconf=1, minsup=0.1, minconf=0.95, max_predicates_per_rule=5, maxsup=0.99,mode=PredicateMode.Dependency):
        self._signals = signals
        self._wsup = wsup
        self._wconf = wconf
        self._minsup = minsup
        self._minconf = minconf
        self._max_predicates_per_rule = max_predicates_per_rule
        self._maxsup = maxsup
        self._mode = mode
        
        self._cont_feats = []
        self._disc_feats = []
        self._cont_signals = []
        self._disc_signals = []
        for signal in self._signals:
            if isinstance(signal, ContinuousSignal):
                self._cont_feats.append(signal.name)
                self._cont_signals.append(signal)
            else:
                self._disc_feats.extend(signal.get_onehot_feature_names())
                self._disc_signals.append(signal)
        
    
    
    def _propose_cutoff_values(self,df,min_samples_leaf):
        cutoffs = {}
        for signal in self._cont_signals:
            cutoffs[signal.name] = []
        if self._mode == PredicateMode.Dependency:
            for signal in self._disc_signals:
                if len(self._cont_feats) > 0:
                    onehot_feats = signal.get_onehot_feature_names()
                    df.loc[:,'tempt_label'] = 0
                    for i in range(len(onehot_feats)):
                        df.loc[df[onehot_feats[i]]==1,'tempt_label'] = i
    
                    xfeats = list(self._cont_feats)
                    x = df[xfeats].values
                    y = df['tempt_label'].values
                    df.drop(columns='tempt_label',inplace=True)
                    model = DecisionTreeClassifier(criterion = "entropy",min_samples_leaf=int(min_samples_leaf))
                    model.fit(x,y)
                    cut_tuples = helper.extract_cutoffs(model.tree_,xfeats)
                    for ct in cut_tuples:
                        cutoffs[ct[0]].append( (ct[1],ct[2]) )
            
            for signal in self._cont_signals:
                yfeat = signal.name
                xfeats = list(self._cont_feats)
                xfeats.remove(yfeat)
            
                x = df[xfeats].values
                y = df[yfeat].values
                y = (y-signal.mean_value)/signal.std_value
                model = DecisionTreeRegressor(min_samples_leaf=int(min_samples_leaf))
            
                model.fit(x,y)
                cut_tuples = helper.extract_cutoffs(model.tree_,xfeats)
                for ct in cut_tuples:
                    cutoffs[ct[0]].append( (ct[1],ct[2]) )
        else:
            if self._mode == PredicateMode.KMeansBins:
                kbins = 'kmeans'
            elif self._mode == PredicateMode.UniformBins:
                kbins = 'uniform'
            for signal in self._cont_signals:
                y = df[signal.name].values
                discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy=kbins)
                y = discretizer.fit_transform(np.reshape(y, (-1, 1)))
                edges = discretizer.bin_edges_[0]
                for i in range(1, len(edges) - 1):
                    cutoffs[signal.name].append((edges[i], 1))
            
        cutoffs = helper.reset_cutoffs(cutoffs,df,min_samples_leaf)
        return cutoffs
    
        
    def train(self, train_df, max_perdicts4rule_mining = 100, max_times4rule_mining = 5, verbose=True):
        """
        learn invariant rules from data
        
        Parameters
        ----------
        train_df : DataFrame
            the training data
        max_perdicts4rule_mining : int
            the max number of predicts can be allowed in a rule mining process (in order to speed up mining process)
        max_times4rule_mining : int
            the max number for the rule mining process, only use for when number of generated predicates is larger than max_perdicts_for_rule_mining
        verbose : Bool
            whether print progress information during training
        
        Returns
        -----------------
        PARsModel
            self
        
        """
        
        min_samples_predicate = int(self._minsup * len(train_df))
        df = train_df.copy()

        cutoffs = self._propose_cutoff_values(df, min_samples_leaf=min_samples_predicate)
#             
        for feat in self._cont_feats:
            if feat not in cutoffs:
                df.drop(columns=feat,inplace=True)
                continue
            
            vals2preserve = [val_pri_pair[0] for val_pri_pair in cutoffs[feat]]
            vals2preserve.sort()
            # print(feat,vals2preserve)           
            for j in range(len(vals2preserve)):
                if j == 0:
                    new_feat = feat + '<' + str(vals2preserve[j])
                    df[new_feat] = 0
                    df.loc[df[feat] < vals2preserve[j], new_feat] = 1
                if j > 0:
                    new_feat = str(vals2preserve[j-1]) + '<=' + feat + '<' + str(vals2preserve[j])
                    df[new_feat] = 0
                    df.loc[(df[feat]<vals2preserve[j]) & (df[feat]>=vals2preserve[j-1]), new_feat] = 1
                if j == len(vals2preserve)-1:
                    new_feat = feat + '>=' + str(vals2preserve[j])
                    df[new_feat] = 0
                    df.loc[df[feat] >= vals2preserve[j], new_feat] = 1
            df.drop(columns=feat,inplace=True)

            
        low_feats = []
        for entry in df.columns:
            support = len(df.loc[df[entry]==1,:])
            if support < min_samples_predicate and entry.find('<') == -1 and entry.find('>') == -1:
                low_feats.append( entry )
        # print('low_feats',low_feats)
        
        start_index = 0
        combine_list = []
        for i in range(1,len(low_feats)):
            df.loc[:,'tempt'] = 0
            for j in range(start_index,i):
                df.loc[df[low_feats[j]]==1, 'tempt'] = 1
            left_support = len(df.loc[df['tempt']==1,:])
            # print(low_feats[i],left_support,min_samples_predicate)
             
            if left_support >= min_samples_predicate:
                df.loc[:,'tempt'] = 0
                for j in range(i+1,len(low_feats)):
                    df.loc[df[low_feats[j]]==1, 'tempt'] = 1
                right_support = len(df.loc[df['tempt']==1,:])
                if right_support >= min_samples_predicate:
                    combine_list.append(low_feats[start_index:i+1])
                    start_index = i+1
                else:
                    combine_list.append(low_feats[start_index:])
                    break
        df = df.drop(columns='tempt',errors='ignore')
        # print('combine list',combine_list)
        
        for entry2combine in combine_list:
            new_feat = entry2combine[0]
            for i in range(1,len(entry2combine)):
                new_feat += ' or ' + entry2combine[i]
            df[new_feat] = 0
            for feat in entry2combine:
                df.loc[df[feat]==1, new_feat] = 1
                df = df.drop(columns=feat)
        
        # print('df shape',df.shape)
        for entry in df.columns:
            support = len(df.loc[df[entry]==1,:])/len(df)
            if support > self._maxsup or support < self._minsup:
                # if verbose:
                #     print('drop',entry, len( df.loc[df[entry]==1,:])/len(df) )
                df = df.drop(columns=entry)
            # else:
            #     if verbose:
            #         print('keep',entry, len( df.loc[df[entry]==1,:])/len(df) )
        
        if verbose:
            print('number of generated predicates:',df.shape[1])
        
        ## start invariant rule mining
        rules, self._item_dict = mine_rules(df, max_len=self._max_predicates_per_rule, theta=self._minsup,
                                                max_perdicts_for_rule_mining = max_perdicts4rule_mining,
                                                min_conf = self._minconf,max_times_for_rule_mining = max_times4rule_mining,verbose=verbose)
        
        self._rules = sorted(rules, key=lambda t: self._wsup*(t.support-self._minsup)/(1-self._minsup) + self._wconf*(t.conf-self._minconf)/(1-self._minconf), reverse=True)
        
        if verbose:
            print('number of generated rules',len(self._rules)+len(self._signals))
        return self
    
    def find_topk_rules(self,df,k=5):
        test_df = df.copy().reset_index(drop=True)
        sigstd = {}
        for signal in self._cont_signals:
            sigstd[signal.name] = signal.std_value
        test_df = helper.parse_predicates(test_df,self._item_dict,sigstd)
         
        topk_rules = []
        
        scores = []
        for signal in self._signals:
            score = test_df.apply(helper.boundary_anomaly_scoring,axis=1,args=(signal,sigstd,)).to_numpy()[0]
            if score > 0:
                topk_rules.append( signal )
                scores.append(score)
        
        temp = sorted(scores,reverse=True)
        if len(topk_rules) >= k:
            res = []
            for i in range(k):
                res.append(topk_rules[scores.index(temp[i])])
            return res
                   
        for rule in self._rules:
            antec_satisfy = True
            conseq_satisfy = True
            
            for item in rule.antec:
                if test_df.loc[0,self._item_dict[item]] != 1:
                    antec_satisfy = False
                    break
            
            if antec_satisfy:
                for item in rule.conseq:
                    if test_df.loc[0,self._item_dict[item]] != 1:
                        conseq_satisfy = False
                        break
                
                if not conseq_satisfy:
                    topk_rules.append( rule )
                    if len(topk_rules) >= k:
                        break
            
        return topk_rules
    
    
    def score_samples(self, df, rules):
        test_df = df.copy()
        sigstd = {}
        for signal in self._cont_signals:
            sigstd[signal.name] = signal.std_value
        test_df = helper.parse_predicates(test_df,self._item_dict,sigstd)
        
        test_df.loc[:,'anomaly_score'] = 0
        for rule in rules:
            test_df.loc[:,'antecedent'] = 1
            test_df.loc[:,'consequent'] = 0
            
            if isinstance(rule,BaseSignal):
                scores = test_df.apply(helper.boundary_anomaly_scoring,axis=1,args=(rule,sigstd,))
                test_df.loc[:,'anomaly_score'] += scores
            else:    
                for item in rule.antec:
                    test_df.loc[test_df[ self._item_dict[item] ]!=1,  'antecedent'] = 0
    
                for item in rule.conseq:
                    test_df.loc[:,  'consequent'] += 1-test_df[ self._item_dict[item] ].values
                    
                scores4rule = np.multiply(test_df['antecedent'].values,test_df['consequent'].values)
                test_df.loc[:,'anomaly_score'] += scores4rule
                
        return test_df.loc[:,'anomaly_score'].values
    
    def predict(self):
        pass
    # def predict(self,df,use_boundary_rules=True):
    #     """
    #     Predict the anomaly scores for data
    #
    #     Parameters
    #     ----------
    #     df : DataFrame
    #         the test data
    #     use_boundary_rules : bool
    #         whether use boundary rules
    #     use_cores : int, default is 1
    #         number of cores to use
    #
    #     Returns
    #     -------
    #     ndarray 
    #         the anomaly scores for each row in df
    #     """
    #
    #     test_df = df.copy()
    #     sigstd = {}
    #     for signal in self._cont_signals:
    #         sigstd[signal.name] = signal.std_value
    #     test_df = helper.parse_predicates(test_df,self._item_dict,sigstd)
    #
    #     exp = AnomalyExplanation()
    #
    #     if use_boundary_rules:
    #         for signal in self._signals:
    #             scores = test_df.apply(helper.boundary_anomaly_scoring,axis=1,args=(signal,sigstd,)).to_numpy()
    #             indices = np.where(scores>0)[0]
    #             # print(scores)
    #             # print(indices)
    #             for i in indices:
    #                 exp.add_record(signal.name, i, self._wsup+self._wconf, helper.boundary_rule(signal),[signal.name])
    #
    #
    #     for i in test_df.index:
    #         # print(i,'/',len(test_df))           
    #         for rule in self._rules:
    #             antec_satisfy = True
    #
    #             for item in rule.antec:
    #                 if test_df.loc[i,self._item_dict[item]] != 1:
    #                     antec_satisfy = False
    #                     break
    #
    #             if antec_satisfy:
    #                 for item in rule.conseq:
    #                     if test_df.loc[i,self._item_dict[item]] != 1:
    #                         feat = helper.extract_feat_from_predicate(self._item_dict[item])
    #                         score = self._wsup*(rule.support-self._minsup)/(1-self._minsup) + self._wconf*(rule.conf-self._minconf)/(1-self._minconf)
    #                         if feat is not None:
    #                             exp.add_record(feat, i, score, str(rule),rule.extract_feats())
    #     return exp
              
    def export_rules(self, filepath):
        """
        export rules to file
        
        Parameters
        ----------
        filepath : string
            the file path
        """
        with open(filepath,'w') as myfile:
            for signal in self._signals:
                myfile.write(helper.boundary_rule(signal) + '\n')
            for rule in self._rules:
                myfile.write(str(rule) + '\n')
            myfile.close()
    
    
    def get_num_rules(self):
        """
        get number of rules
        
        Returns
        -------
        int
            the number of rules
        """
        return len(self._rules)+len(self._signals)
    
    def save_model(self,model_path=None, model_id=None):
        """
        save the model to files
        
        Parameters
        ----------
        model_path : string, default is None
            the target folder whether the model files are saved.
            If None, a tempt folder is created
        model_id : string, default is None
            the id of the model, must be specified when model_path is not None
        """
        
        model_path = super().save_model(model_path, model_id)
        pickle.dump(self._rules,open(model_path+'/rules.pkl','wb'))
        pickle.dump(self._item_dict,open(model_path+'/item_dict.pkl','wb'))
        config_dict = dict(
            wsup = self._wsup,
            wconf = self._wconf, 
            minsup = self._minsup, 
            minconf = self._minconf,
            max_predicates_per_rule = self._max_predicates_per_rule,
            maxsup = self._maxsup
            )
        pickle.dump(config_dict,open(model_path+'/config_dict.pkl','wb'))
    
    def load_model(self,model_path=None, model_id=None):
        """
        load the model from files
        
        Parameters
        ----------
        model_path : string, default is None
            the target folder whether the model files are located
            If None, load models from the tempt folder
        model_id : string, default is None
            the id of the model, must be specified when model_path is not None
            
        Returns
        -------
        PARsModel
            self
        """
        model_path = super().load_model(model_path, model_id)
        self._item_dict = pickle.load(open(model_path+'/item_dict.pkl','rb'))    
        self._rules = pickle.load(open(model_path+'/rules.pkl','rb'))
    
        config_dict = pickle.load(open(model_path+'/config_dict.pkl','rb'))
        self._wsup = config_dict['wsup']
        self._wconf = config_dict['wconf']
        self._minsup = config_dict['minsup']
        self._minconf = config_dict['minconf']
        self._max_predicates_per_rule = config_dict['max_predicates_per_rule']
        self._maxsup = config_dict['maxsup']
        return self