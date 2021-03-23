import numpy as np
import pandas as pd

def getSubjectData(Data,num_variables):
    CData = np.array(Data['CData'])
    sub_data = pd.DataFrame()

    sub_data['probe']        = CData[:,10] # -1: L, 1: R
    sub_data['response']     = CData[:,11] #  1: Ch, 0: N-Ch, 5: response not recorded
    sub_data['RT']           = CData[:,12] #  response time
    # Additional fields :
    sub_data["correct"] =   CData[:,13]
    sub_data["answer"] =   CData[:,14]
    sub_data["probed_Ch"] =  CData[:,17] + CData[:,18]
    sub_data["unprobed_Ch"] =  CData[:,15] + CData[:,16]
    sub_data["unprobed_Ch_L"] = CData[:,15]
    sub_data["unprobed_Ch_R"] = CData[:,16]
    sub_data["probed_Ch_L"] = CData[:,17]
    sub_data["probed_Ch_R"] = CData[:,18]
    sub_data["probe_L"] = CData[:,19]
    sub_data["probe_R"] = CData[:,20]
    sub_data["probL"] = CData[:,21] #probability of left side cue

    # dump mistrials
    indices = np.where(sub_data['response'] != 5)[0].tolist()
    sub_data=sub_data.iloc[indices,:]        
    sub_data["trial_num"]=indices
    sub_data=sub_data.reset_index(drop=True)

    # add previous response as a variable, comment this out if not used
    if num_variables=='prevresp':
        sub_data["prev_resp"]=np.nan
        sub_data.loc[1:,"prev_resp"]=sub_data["response"][:-1].to_numpy()
        # trim off the first trial
        sub_data=sub_data.iloc[1:,:]
        sub_data=sub_data.reset_index(drop=True)

    return sub_data 


def getData(sub_data,num):
    if num=='3':
        dat,weights=getData_3(sub_data)
    elif num=='4':
        dat,weights=getData_4(sub_data)    
    elif num=='prevresp':
        dat,weights=getData_prevresp(sub_data)    
    elif num=='6':
        dat,weights=getData_6(sub_data)    

    return dat, weights    

def getData_3(sub_data):

    
    inputs = dict(
            probed_Ch = np.array(sub_data["probed_Ch"])[:, None],
            unprobed_Ch = np.array(sub_data["unprobed_Ch"])[:, None],
                 )                  
                
    dat = dict(
        
        probed_Ch = np.array(sub_data["probed_Ch"]),
        unprobed_Ch = np.array(sub_data["unprobed_Ch"]),

        correct=np.array(sub_data['correct']),
        answer=np.array(sub_data['answer']),
        inputs = inputs,
        probL=np.array(sub_data['probL']),
        y = np.array(sub_data['response'])
    )
    
    weights = {'bias' : 1, 'probed_Ch' : 1, 'unprobed_Ch':1}

    return dat, weights


def getData_4(sub_data):

    
    inputs = dict(
                probed_Ch_L = np.array(sub_data["probed_Ch_L"])[:, None],
                probed_Ch_R = np.array(sub_data["probed_Ch_R"])[:, None],
                probe_L = np.array(sub_data["probe_L"])[:, None],
                probe_R = np.array(sub_data["probe_R"])[:, None]
            )                  
                
    dat = dict(
        
        probed_Ch_L = np.array(sub_data["probed_Ch_L"]),
        probed_Ch_R = np.array(sub_data["probed_Ch_R"]),
        probe_L = np.array(sub_data["probe_L"]),
        probe_R = np.array(sub_data["probe_R"]),

        correct=np.array(sub_data['correct']),
        answer=np.array(sub_data['answer']),
        inputs = inputs,
        probL=np.array(sub_data['probL']),
        y = np.array(sub_data['response'])
    )
    
    weights = {'probed_Ch_L' : 1, 'probed_Ch_R' : 1, 'probe_L' : 1, 'probe_R' : 1}

    return dat, weights


def getData_6(sub_data):

    
    inputs = dict(
                probed_Ch_L = np.array(sub_data["probed_Ch_L"])[:, None],
                probed_Ch_R = np.array(sub_data["probed_Ch_R"])[:, None],
                unprobed_Ch_L = np.array(sub_data["unprobed_Ch_L"])[:, None],
                unprobed_Ch_R = np.array(sub_data["unprobed_Ch_R"])[:, None],
                probe_L = np.array(sub_data["probe_L"])[:, None],
                probe_R = np.array(sub_data["probe_R"])[:, None]
            )                  
                
    dat = dict(
        
        probed_Ch_L = np.array(sub_data["probed_Ch_L"]),
        probed_Ch_R = np.array(sub_data["probed_Ch_R"]),
        unprobed_Ch_L = np.array(sub_data["unprobed_Ch_L"]),
        unprobed_Ch_R = np.array(sub_data["unprobed_Ch_R"]),
        probe_L = np.array(sub_data["probe_L"]),
        probe_R = np.array(sub_data["probe_R"]),

        correct=np.array(sub_data['correct']),
        answer=np.array(sub_data['answer']),
        inputs = inputs,
        probL=np.array(sub_data['probL']),
        y = np.array(sub_data['response'])
    )
    
    weights = {'probed_Ch_L' : 1, 'probed_Ch_R' : 1, 'unprobed_Ch_L' : 1, 'unprobed_Ch_R' : 1, 'probe_L' : 1, 'probe_R' : 1}

    return dat, weights

def getData_prevresp(sub_data):

    
    inputs = dict(
                probed_Ch_L = np.array(sub_data["probed_Ch_L"])[:, None],
                probed_Ch_R = np.array(sub_data["probed_Ch_R"])[:, None],
                probe_L = np.array(sub_data["probe_L"])[:, None],
                probe_R = np.array(sub_data["probe_R"])[:, None],
                prev_resp = np.array(sub_data["prev_resp"])[:, None]
            )                  
                
    dat = dict(
        
        probed_Ch_L = np.array(sub_data["probed_Ch_L"]),
        probed_Ch_R = np.array(sub_data["probed_Ch_R"]),
        probe_L = np.array(sub_data["probe_L"]),
        probe_R = np.array(sub_data["probe_R"]),
        prev_resp = np.array(sub_data["prev_resp"]),

        correct=np.array(sub_data['correct']),
        answer=np.array(sub_data['answer']),
        inputs = inputs,
        probL=np.array(sub_data['probL']),
        y = np.array(sub_data['response'])
    )

    weights = {'prev_resp' : 1, 'probed_Ch_L' : 1, 'probed_Ch_R' : 1, 'probe_L' : 1, 'probe_R' : 1}
    
    return dat, weights        