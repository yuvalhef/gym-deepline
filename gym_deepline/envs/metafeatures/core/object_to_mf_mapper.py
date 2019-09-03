import itertools

def map_object_to_mf(attr_dict, mf):

    ticketsFeatures = []
    ticketsLabels = []
    ticketsFeaturesLabels = []

    numAttr = {k for k, v in attr_dict.items() if v['type'] == 'numerical'}
    catAttr = {k for k, v in attr_dict.items() if v['type'] == 'categorical'}
    regLabel = {k for k, v in attr_dict.items() if (v['type'] == 'numerical') and ( v['is_target'] == True )}
    classLabel = {k for k, v in attr_dict.items() if (v['type'] == 'categorical') and ( v['is_target'] == True )}

    # MATRIX OBJECTS. Example: count number of rows.
    if mf.get_matrix_applicable() == True:
        raise NotImplementedError

    # Metafunction that can handle both numerical and categorical variables - Missing values
    if mf.get_numerical_arity()==mf.get_categorical_arity():
        ticketsFeatures += [list(catAttr)[i:i + 1] for i in range(0, len(catAttr))]
        ticketsFeatures += [list(numAttr)[i:i + 1] for i in range(0, len(numAttr))]
        ticketsLabels += [list(classLabel)[i:i + 1] for i in range(0, len(classLabel))]
        ticketsLabels += [list(regLabel)[i:i + 1] for i in range(0, len(regLabel))]
        #raise NotImplementedError

    # Classification dataset and metafunction like entropy (categorical arity = 1)
    if (mf.get_numerical_arity()==0) and (mf.get_categorical_arity()==1) and (len(classLabel)!=0):
        ticketsFeatures += [list(catAttr)[i:i + 1] for i in range(0, len(catAttr))]
        ticketsLabels += [list(classLabel)[i:i + 1] for i in range(0, len(classLabel))]
    # Regression dataset without categorical variables and metafunction like entropy
    if (mf.get_numerical_arity()==0) and (mf.get_categorical_arity()==1) and (len(regLabel)!=0) and (len(catAttr)==0):
        pass
    # Regression dataset with one or more categorical variables and metafunction like entropy
    if (mf.get_numerical_arity()==0) and (mf.get_categorical_arity()==1) and (len(regLabel)!=0) and (len(catAttr)>=1):
        ticketsFeatures += [list(catAttr)[i:i + 1] for i in range(0, len(catAttr))]

    # Classification dataset with one or more categorical vars and metafunction like mutual
    # information (categorical arity = 2)
    if (mf.get_numerical_arity()==0) and (mf.get_categorical_arity()==2) and (len(classLabel)!=0) and (len(catAttr)>=1):
        ticketsFeatures += [list(subset) for subset in itertools.combinations(catAttr, 2)]
        if (len(regLabel) >= 2):
            ticketsLabels += [list(subset) for subset in itertools.combinations(classLabel, 2)]
        ticketsFeaturesLabels += list(itertools.product(classLabel, catAttr))
    # Classification dataset without categorical variables and metafunction like mutual information
    if (mf.get_numerical_arity()==0) and (mf.get_categorical_arity()==2) and (len(classLabel)==1) and (len(catAttr)==0):
        pass
    # Multi-label classification dataset without categorical variables
    if (mf.get_numerical_arity()==0) and (mf.get_categorical_arity()==2) and (len(classLabel)>=2) and (len(catAttr)==0):
        ticketsLabels += [list(subset) for subset in itertools.combinations(classLabel, 2)]
    # Regression dataset with two or more categorical variables and metafunction like mutual information
    if (mf.get_numerical_arity()==0) and (mf.get_categorical_arity()==2) and (len(regLabel)!=0) and (len(catAttr)>=2):
        ticketsFeatures += [list(subset) for subset in itertools.combinations(catAttr, 2)]
    # Regression dataset with one or less categorical variables and metafunction like mutual information
    if (mf.get_numerical_arity()==0) and (mf.get_categorical_arity()==2) and (len(regLabel)!=0) and (len(catAttr)<=1):
        pass

    # Regression dataset with metafunction like average
    if (mf.get_numerical_arity()==1) and (mf.get_categorical_arity()==0) and (len(regLabel)!=0):
        ticketsFeatures += [list(numAttr)[i:i + 1] for i in range(0, len(numAttr))]
        ticketsLabels += [list(regLabel)[i:i + 1] for i in range(0, len(regLabel))]
    # Classification dataset without numerical variables and metafunction like average
    if (mf.get_numerical_arity()==1) and (mf.get_categorical_arity()==0) and (len(classLabel)!=0) and (len(numAttr)==0):
        pass
    # Classification dataset with one or more numerical variables and metafunction like average
    if (mf.get_numerical_arity()==1) and (mf.get_categorical_arity()==0) and (len(classLabel)!=0) and (len(numAttr)>=1):
        ticketsFeatures += [list(numAttr)[i:i + 1] for i in range(0, len(numAttr))]

    # Classification dataset with two or more numerical vars and metafunction like correlation
    if (mf.get_numerical_arity()==2) and (mf.get_categorical_arity()==0) and (len(classLabel)!=0) and (len(numAttr)>=2):
        ticketsFeatures += [list(subset) for subset in itertools.combinations(numAttr, 2)]
    # Classification dataset with one or less numerical vars and metafunction like correlation
    if (mf.get_numerical_arity()==2) and (mf.get_categorical_arity()==0) and (len(classLabel)!=0) and (len(numAttr)<=1):
        pass
    # Regression dataset with more than one numerical variables and metafunction like correlation
    if (mf.get_numerical_arity()==2) and (mf.get_categorical_arity()==0) and (len(regLabel)!=0) and (len(numAttr)>=1):
        ticketsFeatures += [list(subset) for subset in itertools.combinations(numAttr, 2)]
        if (len(regLabel) >= 2):
            ticketsLabels += [list(subset) for subset in itertools.combinations(regLabel, 2)]
        ticketsFeaturesLabels += list(itertools.product(regLabel, numAttr))
    # Regression dataset without numerical variables and metafunction like correlation
    if (mf.get_numerical_arity()==2) and (mf.get_categorical_arity()==0) and (len(regLabel)==1) and (len(numAttr)==0):
        pass
    # Multi target regression dataset without numerical variables and metafunction like correlation
    if (mf.get_numerical_arity()==2) and (mf.get_categorical_arity()==0) and (len(regLabel)>=2) and (len(numAttr)==0):
        ticketsLabels += [list(subset) for subset in itertools.combinations(regLabel, 2)]

    return ticketsFeatures, ticketsLabels, ticketsFeaturesLabels


