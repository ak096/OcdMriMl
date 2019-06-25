from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import pandas as pd


def scale(frame):

    columns = frame.columns
    index = frame.index
    frame_norms = []
    scalers = [StandardScaler()] #,MinMaxScaler(), RobustScaler()]

    for s in scalers:
        frame_norms.append(pd.DataFrame(index=index, columns=columns, data=s.fit_transform(frame)))

    return frame_norms[0].round(3), scalers[0]


def test_set_scale(frame, scalers):

    columns = frame.columns
    index = frame.index
    frame_norm = []
    try:
        for s in scalers:
            frame_norm.append(pd.DataFrame(index=index, columns=columns, data=s.transform(frame)))
    except TypeError as te:
        frame_norm.append(pd.DataFrame(index=index, columns=columns, data=scalers.transform(frame)))

    return frame_norm[0].round(3)
