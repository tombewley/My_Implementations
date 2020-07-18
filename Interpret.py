import numpy as np


def factualExplanation(df, leaf_uid, better_feature_names=None, ignore_features=set(), print_out=True):
    """Given a leaf UID, print out a factual explanation of the decision in terms of feature ranges."""
        
    leaf = df.loc[leaf_uid]
    feature_names = set([f[:-2] for f in list(df) if '>' in f])
    ranges = {}
    for f in feature_names - ignore_features:   
        if better_feature_names: f_better = better_feature_names[f] 
        else: f_better = f
        l, u = leaf[f+' >'], leaf[f+' <'] 
        ranges[f_better] = (l,u)
            
        # DON'T ISE THIS CODE BELOW, BECAUSE IT'S MISLEADING - IT SUGGESTS A LARGER BOUNDING BOX THAN IS ACTUALLY THE CASE.

        # if generalise:
        #     # Seek to extend bounds by looking at neighbouring leaves.
        #     for bound, symbol, other_symbol in [(l,' >',' <'),(u,' <', ' >')]:
        #         if abs(bound) != np.inf:

        #             # Find neighbouring leaves of the same class along feature dimension f, in the given direction.
        #             neighbours = df.loc[(df[f+other_symbol]==bound) & (df['class']==leaf['class'])].copy()
        #             if len(neighbours) > 0:
                        
        #                 # Check if the input instance X would lie within each neighbour if it weren't for the current feature f.
        #                 neighbours.loc[:,'otherwise'] = neighbours.apply(lambda row: sum([row[ff+' >'] > X[ii] or row[ff+' <'] < X[ii] for ii, ff in enumerate(feature_names) if ff != f])==0, axis=1)
                        
        #                 neighbour_index = list(neighbours[neighbours['otherwise'] == 1].index)
        #                 assert len(neighbour_index) <= 1 # At most one neighbour should pass this test.
        #                 if len(neighbour_index) == 1:
        #                     neighbour = neighbours.loc[neighbour_index[0]]

        #                     # Replace bound with the one from the passing neighbour.
        #                     if symbol == ' >':   l = neighbour[f+symbol]
        #                     elif symbol == ' <': u = neighbour[f+symbol]

        # Print out the ranges for this feature.
        if print_out and not (l == -np.inf and u == np.inf):
            print('    {} is {}'.format(f_better, 
                                        ('>= {}'.format(round(l,30)) if u == np.inf else (
                                         '< {}'.format(round(u,30)) if l == -np.inf else
                                         'between {} and {}'.format(round(l,30), round(u,30))))))
    return ranges


def actionBasedCounterfactual(df, X, y_foil, better_feature_names=None):
    """Given an input vector and a foil class, print out the set of minimal counterfactuals."""

    # Identify the regions of the feature space for which the foil action is taken.
    df_foil = df.loc[df['class']==y_foil].copy()

    # For each feature, compute the change required to move into each desired region and add a column containing this value.
    feature_names = [f[:-2] for f in list(df) if '>' in f]
    for i, f in enumerate(feature_names):
        df_foil.loc[:,'Δ ' +f] = df_foil.apply(lambda row: row[f+' >'] - X[i] if row[f+' >'] > X[i] else (row[f+' <'] - X[i] if X[i] >= row[f+' <'] else 0.), axis=1)

    # Compute some summary statistics.
    df_foil.loc[:,'del sum'] = df_foil.apply(lambda row: sum([abs(row[col]) for col in df_foil.columns if 'Δ' in col]), axis=1) # L1 norm of Δs.
    df_foil.loc[:,'del count'] = df_foil.apply(lambda row: sum([row[col]!=0 for col in df_foil.columns if 'Δ' in col]), axis=1) # Number of nonzero Δs.

    # Find the best regions to move to. 
    # *** Multiple ways of framing this *** but here find rows with del count < a threshold, sort by del sum, and return all deltas of the first row.
    thresh = 1
    while True:
        df_matches = df_foil.loc[df_foil['del count'] <= thresh].sort_values(by='del sum')
        if df_matches.shape[0] > 0:
            deltas = {k[2:]:v for k,v in df_matches.to_dict().items() if 'Δ' in k}
            break
        thresh += 1

    used_feature_combinations = []
    for n, uid in enumerate(df_matches.index):
        counterfactuals = []
        feature_combination = set()
        for i, f in enumerate(feature_names):
            if deltas[f][uid] != 0: 
                feature_combination.add(f)
                counterfactuals.append('{} were {} instead of {}'.format((better_feature_names[f] if better_feature_names else f),  
                                                                                                       round(X[i]+deltas[f][uid], 6), 
                                                                                                       round(X[i], 6)))                                                                                                                         
        # Only continue if this feature combination hasn't been seen already.
        # Note that df_matches is ordered so closer counterfactuals come first.
        if feature_combination not in used_feature_combinations:
            # Print out the set of conditions for this counterfactual.
            print(' {} {} (UID = {})'.format(('or' if n > 0 else '  '),' and '.join(counterfactuals), uid))
            used_feature_combinations.append(feature_combination)


def leafBasedCounterfactual(df, X, leaf_uid_foil, better_feature_names=None):
    """Given an input vector and a foil leaf, print out the counterfactual."""

    leaf_foil = df.loc[leaf_uid_foil].to_dict()
    feature_names = [f[:-2] for f in list(df) if '>' in f]
    counterfactuals = []
    trigger_features = set()
    for i, f in enumerate(feature_names):
        if leaf_foil[f+' >'] > X[i]:    
            trigger_features.add(f)
            counterfactuals.append('{} dropped below {} (={})'.format((better_feature_names[f] if better_feature_names else f),
                                                                                                  leaf_foil[f+' >'], X[i]))
        elif X[i] >= leaf_foil[f+' <']: 
            trigger_features.add(f)
            counterfactuals.append('{} matched or exceeded {} (={})'.format((better_feature_names[f] if better_feature_names else f),
                                                                                             leaf_foil[f+' <'], X[i]))
    print('    {}'.format(' and '.join(counterfactuals)))

    return trigger_features