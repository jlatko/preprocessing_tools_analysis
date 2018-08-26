
#  BINNING
params = [
    {  # BASELINE
        'clipper': [None],
        'binner': [None],
        'binner2': [None],
        'boxcox': [None],
        'scaler': [None],
        'predictor': predictors
    },
    {  # BINNER
        'clipper': [None],
        'binner2': [
            CustomBinner(BINNER_CONFIG_PRUD)
        ],
        'binner2__nan': [True, False],
        'binner2__drop': [True, False],
        'binner': [None],
        'boxcox': [None],
        'scaler': [None],
        'predictor': predictors
    },
    {  # BINNER Binary
        'clipper': [None],
        'binner2': [
            CustomBinaryBinner(BINARY_BINNER_CONFIG_PRUD)
        ],
        'binner2__nan': [True, False],
        'binner2__drop': [True, False],
        'binner': [None],
        'boxcox': [None],
        'scaler': [None],
        'predictor': predictors
    },
]

def get_class(params):
    if not params['binner2']:
        return 'baseline', 'basic'
    elif isinstance(params['binner2'], CustomBinner):
        model_name = 'binner'
    elif isinstance(params['binner2'], CustomBinaryBinner):
        model_name = 'binnary binner'
    else:
        model_name = 'other?'

    labels = []
    if params['binner2__nan']:
        labels.append('nan')
    if params['binner2__drop']:
        labels.append('drop')
    if labels:
        label = ', '.join(labels)
    else:
        label = 'basic'
    return model_name, label

#  ======================

# OUTLIERS


BINNER_CONFIG = [{col: {'bins': 3} for col in continuous + discrete},
                 {col: {'bins': 5} for col in continuous + discrete},
                 {col: {'bins': 7} for col in continuous + discrete},
                 {col: {'values': [train[col].max()]} for col in continuous + discrete}]

def get_class(params):
    if not params['binner']:
        return 'baseline', 'basic'
    elif isinstance(params['binner'], CustomBinner):
        model_name = 'binner'
    elif isinstance(params['binner'], CustomBinaryBinner):
        model_name = 'binnary binner'
    else:
        model_name = 'other?'
    conf = params['binner__configuration']
    if 'values' in next(iter(conf.values())):
        label = 'max flag'
    else:
        label = 'bins ' + str(next(iter(conf.values()))['bins'])
    return model_name, label
    # params = [
    #     {  # BASELINE
    #         'onehot': [one_hot],
    #         'clipper': [None],
    #         'binner': [None],
    #         'binner2': [None],
    #         'boxcox': [None],
    #         'scaler': [None],
    #         'predictor': predictors
    #     },
    #     {  # Binner
    #         'onehot': [one_hot],
    #         'clipper': [None],
    #         'binner':  [CustomBinaryBinner({}), CustomBinner({})],
    #         'binner__configuration': [c for c in BINNER_CONFIG],
    #         'binner2': [None],
    #         'boxcox': [None],
    #         'scaler': [None],
    #         'predictor': predictors
    #     },
    # ]


# =========   SCALING  =====================
def get_class(params):
    model_names = []
    if params['scaler'] and isinstance(params['scaler'], StandardScaler):
        model_names.append('standard')
    if params['scaler'] and isinstance(params['scaler'], RobustScaler):
        model_names.append('robust')
    if params['boxcox']:
        model_names.append('boxcox')
    if not model_names:
        model_name =  'base'
    else:
        model_name = ', '.join(model_names)

    if params['clipper']:
        label = 'clipped'
    else:
        label = 'not clipped'
    return model_name, label

    params = [
        {  # BASELINE
            'onehot': [one_hot],
            'clipper': [None, OutliersClipper(continuous)],
            'binner': [None],
            'binner2': [None],
            'boxcox': [None, BoxCoxTransformer(BOX_COX)],
            'scaler': [None, StandardScaler(), RobustScaler()],
            'predictor': predictors
        },
    ]


#  ========== Imputing

def get_class(params):
    model_names = []
    imputer1 = params['simple_imputer']
    if imputer1:
        if imputer1.mean:
            model_names.append('median')
        if imputer1.median:
            model_names.append('mean')
        if imputer1.nan_flag:
            model_names.append('nan')
    else:
        model_names.append('zero')
    model_name = ', '.join(model_names)

    label = type(params['predictor']).__name__
    return model_name, label

    params = [
        {  # BASELINE
            'onehot': [one_hot],
            'clipper': [None],
            'binner': [None],
            'binner2': [None],
            'simple_imputer': [
                FillNaTransformer(),
                # FillNaTransformer(mean=missing),
                # FillNaTransformer(median=missing)
            ],
            'simple_imputer__nan_flag': [[], missing],
            'main_imputer': [None],
            'boxcox': [BoxCoxTransformer(BOX_COX)],
            'scaler': [StandardScaler()],
            'predictor': predictors
        },