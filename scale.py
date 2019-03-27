from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import pandas as pd


def scale(frame):

    columns = frame.columns
    index = frame.index
    frame_norms = []
    # ss = StandardScaler()
    # ms = MinMaxScaler()
    # rs = RobustScaler()
    scalers = [StandardScaler(), MinMaxScaler(), RobustScaler()]

    for s in scalers:
        frame_norms.append(pd.DataFrame(index=index, columns=columns, data=s.fit_transform(frame)))

    return frame_norms, scalers


def testSet_scale(frame, scalers):

    columns = frame.columns
    index = frame.index
    frame_norms = []

    for s in scalers:
        frame_norms.append(pd.DataFrame(index=index, columns=columns, data=s.transform(frame)))

    return frame_norms
