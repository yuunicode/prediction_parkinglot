def param(lot, key):

    if lot == 'sichung':
        best_params_total =  {'max_depth': 6, 'learning_rate': 0.5103759746797008, 'min_child_weight': 9, 'gamma': 0.9497607765077224, 'subsample': 0.7492535061512953, 'colsample_bytree': 0.6123983227654238, 'reg_alpha': 0.6792251459038496, 'reg_lambda': 0.9042122848795835, 'random_state': 9}
        best_params_half = {'max_depth': 6, 'learning_rate': 0.5060482481278166, 'min_child_weight': 9, 'gamma': 0.9553314628600317, 'subsample': 0.6272935021426335, 'colsample_bytree': 0.6896677286664026, 'reg_alpha': 0.6514559762349507, 'reg_lambda': 0.8679114048519582, 'random_state': 6}
        best_params_quarter =  {'max_depth': 5, 'learning_rate': 0.5076561785817914, 'min_child_weight': 10, 'gamma': 0.836772176840265, 'subsample': 0.8138735851567991, 'colsample_bytree': 0.5060646869411305, 'reg_alpha': 0.81963769548222, 'reg_lambda': 0.6054281554790452, 'random_state': 37}
        best_params_pizza =   {'max_depth': 4, 'learning_rate': 0.5541397132604415, 'min_child_weight': 9, 'gamma': 0.8980729313012417, 'subsample': 0.7214313501525216, 'colsample_bytree': 0.9928846697866287, 'reg_alpha': 0.7458284164517559, 'reg_lambda': 0.8565079021733094, 'random_state': 44}
        param_dict = {'total': best_params_total, 'half': best_params_half, 'quarter': best_params_quarter, 'one_eight': best_params_pizza}
        return param_dict[key]
    
    if lot == 'unam':
        best_params_total ={'max_depth': 6, 'learning_rate': 0.5216311595420314, 'min_child_weight': 6, 'gamma': 0.8906909241058741, 'subsample': 0.6374834534154609, 'colsample_bytree': 0.9046187948608263, 'reg_alpha': 0.15527792162057205, 'reg_lambda': 0.02398458109781898, 'random_state': 36}
        best_params_half = {'max_depth': 6, 'learning_rate': 0.698126824046296, 'min_child_weight': 8, 'gamma': 0.14007763037539986, 'subsample': 0.6630288889031135, 'colsample_bytree': 0.6692853083580174, 'reg_alpha': 0.022326699644974468, 'reg_lambda': 0.5129263303278474, 'random_state': 13}
        best_params_quarter =   {'max_depth': 6, 'learning_rate': 0.7117457427548044, 'min_child_weight': 5, 'gamma': 0.18367404296250156, 'subsample': 0.9900089871260187, 'colsample_bytree': 0.9972379762730545, 'reg_alpha': 0.5768604823245148, 'reg_lambda': 0.2776826993673065, 'random_state': 38}
        best_params_pizza =  {'max_depth': 6, 'learning_rate': 0.9040822261097736, 'min_child_weight': 5, 'gamma': 0.17936676340397656, 'subsample': 0.9918085598621839, 'colsample_bytree': 0.951616618036657, 'reg_alpha': 0.5870202479958593, 'reg_lambda': 0.9965351009618184, 'random_state': 38}
        param_dict = {'total': best_params_total, 'half': best_params_half, 'quarter': best_params_quarter, 'one_eight': best_params_pizza}
        return param_dict[key]

  
    if lot == 'ohsaek1':
        best_params_total = {'max_depth': 11, 'learning_rate': 0.2884408693293081, 'min_child_weight': 3, 'gamma': 0.1846181555079997, 'subsample': 0.8981951915229429, 'colsample_bytree': 0.8621489522700373, 'reg_alpha': 0.997210003626078, 'reg_lambda': 0.1238831228272474, 'random_state': 25}
        best_params_half =  {'max_depth': 7, 'learning_rate': 0.23038875658379102, 'min_child_weight': 6, 'gamma': 0.5205410360294325, 'subsample': 0.7152018661817188, 'colsample_bytree': 0.8771099312796024, 'reg_alpha': 0.2815119992162783, 'reg_lambda': 0.718706960073288, 'random_state': 36}
        best_params_quarter ={'max_depth': 5, 'learning_rate': 0.26380002593575674, 'min_child_weight': 6, 'gamma': 0.9368835013079411, 'subsample': 0.23206675867758647, 'colsample_bytree': 0.6790013613287827, 'reg_alpha': 0.21208241464969965, 'reg_lambda': 0.5820867812752961, 'random_state': 25}
        # best_params_pizza =  

        ohsaek_total = {'max_depth': 9, 'learning_rate': 0.6180213469448876, 'min_child_weight': 7, 'gamma': 0.7825631769789023, 'subsample': 0.7125433981764251, 'colsample_bytree': 0.5990503286081345, 'reg_alpha': 0.4308354766016503, 'reg_lambda': 0.3887711286662512, 'random_state': 2}
        ohsaek_half = {'max_depth': 9, 'learning_rate': 0.6041503898718803, 'min_child_weight': 6, 'gamma': 0.8744019412693059, 'subsample': 0.6489549086118133, 'colsample_bytree': 0.6123983227654238, 'reg_alpha': 0.3594188594278872, 'reg_lambda': 0.5281592136596875, 'random_state': 6}
        ohsaek_quarter = {'max_depth': 5, 'learning_rate': 0.6701440206935806, 'min_child_weight': 7, 'gamma': 0.6973358237776228, 'subsample': 0.6132864847769419, 'colsample_bytree': 0.794492488970654, 'reg_alpha': 0.5801522106393783, 'reg_lambda': 0.515972823951657, 'random_state': 8}
        ohsaek_pizza = {'max_depth': 9, 'learning_rate': 0.6372796739174249, 'min_child_weight': 5, 'gamma': 0.6300387892964403, 'subsample': 0.9824768070504256, 'colsample_bytree': 0.8042204719157238, 'reg_alpha': 0.5697555945955451, 'reg_lambda': 0.30315486798759617, 'random_state': 13}

        param_dict = {'total': ohsaek_total, 'half': ohsaek_half, 'quarter': ohsaek_quarter, 'one_eight': ohsaek_pizza}
        return param_dict[key]
    
    if lot == 'ohsaek2':

        best_params_total = {'max_depth': 19, 'learning_rate': 0.7681335075206372, 'min_child_weight': 6, 'gamma': 0.4646045422235497, 'subsample': 0.8252371224455493, 'colsample_bytree': 0.6841742128784865, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'random_state': 11}
        best_params_half = {'max_depth': 18, 'learning_rate': 0.7450104267317552, 'min_child_weight': 7, 'gamma': 0.3626884613535699, 'subsample': 0.9424418857590604, 'colsample_bytree': 0.7431182050579215, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'random_state': 17}
        best_params_quarter = {'max_depth': 11, 'learning_rate': 0.5018424640655876, 'min_child_weight': 7, 'gamma': 0.4501019837630806, 'subsample': 0.7574352283386463, 'colsample_bytree': 0.926292262372693, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'random_state': 4}
        best_params_pizza = {'max_depth': 18, 'learning_rate': 0.7450104267317552, 'min_child_weight': 7, 'gamma': 0.3626884613535699, 'subsample': 0.9424418857590604, 'colsample_bytree': 0.7431182050579215, 'reg_alpha': 0.1, 'reg_lambda': 0.1, 'random_state': 17}

        param_dict = {'total': best_params_total, 'half': best_params_half, 'quarter': best_params_quarter, 'one_eight': best_params_pizza}
        return param_dict[key]
