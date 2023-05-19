import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
from .rule import Rule


def _mining_fp(data, max_len, theta,min_conf,index_dict):
    for entry in data:
        data.loc[data[entry]==1, entry] = index_dict[entry]
    df_list = data.values.tolist()
    dataset = []
    for datalist in df_list:
        temptlist = filter(lambda a: a != 0, datalist)
        numbers = list(temptlist)
        dataset.append(numbers)
            
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    frequent = fpgrowth(df, min_support=theta,use_colnames=True,max_len=int(max_len))
      
    df = association_rules(frequent,metric='confidence',min_threshold=min_conf)
    rules = []
    for i in range(len(df)):
        antecedents = df.loc[i,'antecedents']
        consequents = df.loc[i,'consequents']
        confidence = df.loc[i,'confidence']
        support = df.loc[i,'support']
        rule = Rule(antecedents,consequents,confidence,support)
        rules.append(rule)
    return rules


def _multi_fpgrow(df, max_len, theta,min_conf,index_dict,max_perdicts_for_rule_mining, max_times_for_rule_mining,verbose):
    rules_all = []
    for i in range(max_times_for_rule_mining):
        if verbose:
            print('Start rule ming process:',i+1,"/",max_times_for_rule_mining)
        data = df.sample(n=max_perdicts_for_rule_mining,axis='columns')
        rules = _mining_fp(data,max_len,theta,min_conf,index_dict)
        rules_all.extend(rules)
    return rules_all
    
    
def _filter_rules(rules):
    rule_str_set = set()
    rules2keep = []
    for rule in rules:
        if len(rule.conseq)==1 and str(rule) not in rule_str_set:
            rules2keep.append(rule)
            rule_str_set.add(str(rule))
    
    rules2keep = sorted(rules2keep, key=lambda t: t.size())
    
    final_rules = []
    sup_rule_dict = {}
    for rule in rules2keep: 
        candidate_rules = sup_rule_dict.get(rule.support, [])
        dup = False
        for exrule in candidate_rules:
            if set(exrule.conseq) == set(rule.conseq) and set(exrule.antec).issubset( set(rule.antec) ):
                dup = True
                break
        if not dup:
            final_rules.append(rule)
            candidate_rules.append(rule)
            sup_rule_dict[rule.support] = candidate_rules
            
    return final_rules

def mine_rules(df, max_len, theta, min_conf, max_perdicts_for_rule_mining, max_times_for_rule_mining,verbose):
    index_dict = {}
    item_dict = {}
    index = 100
    for entry in df:
        index_dict[entry] = index
        item_dict[index] = entry
        index += 1
    
    if df.shape[1] <= max_perdicts_for_rule_mining or max_times_for_rule_mining<=1:
        data = df.copy()
        rules = _mining_fp(data, max_len, theta,min_conf,index_dict)
    else:
        rules = _multi_fpgrow(df, max_len, theta,min_conf,index_dict,max_perdicts_for_rule_mining, max_times_for_rule_mining,verbose)

    
    for rule in rules:
        rule.set_itemdict(item_dict)
        
    return _filter_rules(rules), item_dict