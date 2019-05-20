import pickle
import gbl
import os


def save_data(data, reg_models=[], clr_models=[]):

    if data == 't':
        trm = open('trm.pkl', 'ab')
        tcm = open('tcm.pkl', 'ab')
        pickle.dump(reg_models, trm, -1)
        pickle.dump(clr_models, tcm, -1)
        trm.close()
        tcm.close()
    elif data == 'h':
        hrm = open('hrm.pkl', 'ab')
        hcm = open('hcm.pkl', 'ab')
        pickle.dump(reg_models, hrm, -1)
        pickle.dump(clr_models, hcm, -1)
        hrm.close()
        hcm.close()
    elif data == 'b':
        brm = open('brm.pkl', 'ab')
        bcm = open('bcm.pkl', 'ab')
        pickle.dump(reg_models, brm, -1)
        pickle.dump(clr_models, bcm, -1)
        brm.close()
        bcm.close()
    elif data == 'tfn':
        tfn = open('tfn.pkl', 'wb')
        pickle.dump(gbl.t_frame_perNorm_list, tfn, -1)
        tfn.close()
    elif data == 'itr':
        itr = open('itr.pkl', 'wb')
        pickle.dump(gbl.iteration, itr, -1)
        itr.close()
    return


def pkl_loader(file):

    print("LOADING PICKLE FILE : %s" % file.name)
    model = []
    while True:
        try:
            model += pickle.load(file)
        except EOFError:
            break
    if not model:
        print("DATA FROM PICKLE FILE EMPTY")
    return model


def load_data():

    try:
        itr = open('itr.pkl', 'rb')
        tfn = open('tfn.pkl', 'rb')
        hrm = open('hrm.pkl', 'rb')
        hcm = open('hcm.pkl', 'rb')
        brm = open('brm.pkl', 'rb')
        bcm = open('bcm.pkl', 'rb')
        trm = open('trm.pkl', 'rb')
        tcm = open('tcm.pkl', 'rb')

        gbl.iteration = pickle.load(itr)
        print("LOADED PICKLE FILE : itr.pkl")
        gbl.t_frame_perNorm_list = pickle.load(tfn)
        print("LOADED PICKLE FILE : tfn.pkl")
        gbl.hoexter_reg_models_all = pkl_loader(hrm)

        gbl.hoexter_clr_models_all = pkl_loader(hcm)

        gbl.boedhoe_reg_models_all = pkl_loader(brm)

        gbl.boedhoe_clr_models_all = pkl_loader(bcm)

        gbl.t_reg_models_all = pkl_loader(trm)

        gbl.t_clr_models_all = pkl_loader(tcm)

        itr.close()
        tfn.close()
        hrm.close()
        hcm.close()
        brm.close()
        bcm.close()
        trm.close()
        tcm.close()

    except (OSError, IOError):
        pass

    return


def remove_data():

    try:
        for file in os.listdir():
            if file.endswith('.pkl'):
                os.remove(file)
    except OSError:
        pass

    return


