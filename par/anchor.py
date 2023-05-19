class Anchor(object):
    '''
    An anchor rule
    '''
    def __init__(self, exp):
        '''
        Constructor
        '''
        self._exp = exp
    
    def __str__(self):
        print(len(self._exp.names()),'dd')
        strRule = ' and '.join(self._exp.names()) + ' ---> abnormal'
        return strRule
    
    def extract_feats(self):
        antec_feats = []
        for item in self._exp.names():
            if item.find(' <= ') != -1 and item.find(' < ') != -1:
                lpos = item.find(' < ')
                rpos = item.find(' <= ')
                feat = item[lpos+3:rpos]
            elif item.find(' <= ') != -1:
                pos = item.find(' <= ')
                feat = item[:pos]
            elif item.find(' < ') != -1:
                pos = item.find(' < ')
                feat = item[:pos]
            elif item.find(' >= ') != -1:
                pos = item.find(' >= ')
                feat = item[:pos]
            elif item.find(' > ') != -1:
                pos = item.find(' > ')
                feat = item[:pos]
            else:
                pos = item.find(' = ')
                feat = item[:pos]
            antec_feats.append(feat)
        return antec_feats
    
    def classify_samples(self, df):
        test_df = df.copy()
        test_df.loc[:,'satisfy'] = 1
        
        for item in self._exp.names():
            if item.find(' <= ') != -1 and item.find(' < ') != -1:
                lpos = item.find(' < ')
                rpos = item.find(' <= ')
                feat = item[lpos+3:rpos]
                lval = float(item[:lpos])
                rval = float(item[rpos+3:])
                test_df.loc[ (test_df[feat]<=lval) | (test_df[feat]>rval),'satisfy'] = 0
            elif item.find(' <= ') != -1:
                pos = item.find(' <= ')
                feat = item[:pos]
                val = float(item[pos+3:])
                test_df.loc[test_df[feat]>val,'satisfy'] = 0
            elif item.find(' < ') != -1:
                pos = item.find(' < ')
                feat = item[:pos]
                val = float(item[pos+2:])
                test_df.loc[test_df[feat]>=val,'satisfy'] = 0
            elif item.find(' >= ') != -1:
                pos = item.find(' >= ')
                feat = item[:pos]
                val = float(item[pos+3:])
                test_df.loc[test_df[feat]<val,'satisfy'] = 0
            elif item.find(' > ') != -1:
                pos = item.find(' > ')
                feat = item[:pos]
                val = float(item[pos+2:])
                test_df.loc[test_df[feat]<=val,'satisfy'] = 0
            else:
                pos = item.find(' = ')
                feat = item[:pos]
                val = float(item[pos+2:])
                test_df.loc[test_df[feat]!=val,'satisfy'] = 0
            # print(feat,val)
        return test_df['satisfy'].values