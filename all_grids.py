
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
