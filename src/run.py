import sys
import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm


def make_in_out(start_date, end_date, waterlevel, tidelevel, rainfall, waterlevel_stations):
    stations = set(waterlevel_stations[waterlevel_stations['評価対象']==1]['観測所名称'])
    in_all_data = {}
    print('\nProcessing waterlevel data')
    for data in waterlevel.groupby('date'):
        day = data[0]
        if day < start_date:
            pass
        elif (day >= start_date) and (day <= end_date):
            data_dict = data[1].to_dict('records')
            in_data = []
            for d in data_dict:
                for k in d.keys():
                    if k not in ('date', 'station', 'river'):
                        in_data.append({'station':d['station'], 'river':d['river'], 'hour':int(k.split(':')[0]), 'value':d[k]})
            in_all_data[day] = {}
            in_all_data[day]['date'] = day
            in_all_data[day]['stations'] = stations
            in_all_data[day]['waterlevel'] = in_data
        elif day > end_date:
            break
    print('done')
    print('Processing rainfall data')
    for data in rainfall.groupby('date'):
        day = data[0]
        if day < start_date:
            pass
        elif (day >= start_date) and (day <= end_date):
            data_dict = data[1].to_dict('records')
            in_data = []
            for d in data_dict:
                for k in d.keys():
                    if k not in ('date', 'station', 'city'):
                        in_data.append({'station':d['station'], 'city':d['city'], 'hour':int(k.split(':')[0]), 'value':d[k]})
            in_all_data[day]['rainfall'] = in_data
        elif day > end_date:
            break
    print('done')
    print('Processing tidelevel data')
    for data in tidelevel.groupby('date'):
        day = data[0]
        if day < start_date:
            pass
        elif (day >= start_date) and (day <= end_date):
            data_dict = data[1].to_dict('records')
            in_data = []
            for d in data_dict:
                for k in d.keys():
                    if k not in ('date', 'station', 'city'):
                        in_data.append({'station':d['station'], 'city':d['city'], 'hour':int(k.split(':')[0]), 'value':d[k]})
            in_all_data[day]['tidelevel'] = in_data
        elif day > end_date:
            break
    print('done')
    start_date_out = start_date + 1
    end_date_out = end_date + 1

    print('Making output data')
    out_all_data = []
    for data in waterlevel.groupby('date'):
        day = data[0]
        if day < start_date_out:
            pass
        elif (day >= start_date_out) and (day <= end_date_out):
            data_dict = data[1].to_dict('records')
            out_data = []
            for d in data_dict:
                if d['station'] in stations:
                    for k in d.keys():
                        if k not in ('date', 'station', 'river'):
                            out_data.append({'date':d['date'], 'station':d['station'], 'hour':int(k.split(':')[0]), 'value':d[k]})
            out_all_data += out_data
        elif day > end_date_out:
            break
    out_all_data = pd.DataFrame(out_all_data)
    out_all_data = out_all_data[(~out_all_data['value'].isin(['M', '*', '-', '--', '**']))&(~out_all_data['value'].isna())]
    out_all_data['value'] = out_all_data['value'].astype(float)
    print('done')
    return in_all_data, out_all_data


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--exec-path', help = '/path/to/submit/src')
    parser.add_argument('--data-dir', help = '/path/to/train')
    parser.add_argument('--start-date', default = 0, type = int, help='start date')
    parser.add_argument('--end-date', default = 9, type = int, help='end date')
    args = parser.parse_args()

    return args


def main(exec_path = '', model_path = '', data_dir = '', start_date = '', end_date = ''):
    
    # parse the arguments

    if (exec_path == '') & (model_path == '') & (data_dir == '') & (start_date == '') & (end_date == ''):
        args = parse_args()
        exec_path = os.path.abspath(args.exec_path)
        model_path = os.path.abspath(args.model_path)
        data_dir = os.path.abspath(args.data_dir)
        start_date = args.start_date
        end_date = args.end_date

    print('\nstart date: {}, end date:{}'.format(start_date, end_date))

    # load the input data and output data
    waterlevel = pd.read_csv(os.path.join(data_dir, 'waterlevel', 'data.csv'))
    tidelevel = pd.read_csv(os.path.join(data_dir, 'tidelevel', 'data.csv'))
    rainfall = pd.read_csv(os.path.join(data_dir, 'rainfall', 'data.csv'))
    waterlevel_stations = pd.read_csv(os.path.join(data_dir, 'waterlevel', 'stations.csv'))
    tidelevel_stations = pd.read_csv(os.path.join(data_dir, 'tidelevel', 'stations.csv'))
    rainfall_stations = pd.read_csv(os.path.join(data_dir, 'rainfall', 'stations.csv'))
    show_dataframe(waterlevel, '水位観測データ')
    show_dataframe(tidelevel, '潮位観測データ')
    show_dataframe(rainfall, '雨量観測データ')
    show_dataframe(waterlevel_stations, '水位観測データ所')
    show_dataframe(tidelevel_stations, '潮位観測データ所')
    show_dataframe(rainfall_stations, '雨量観測データ所')


    
    # change the working directory
    os.chdir(exec_path)
    cwd = os.getcwd()
    print('\nMoved to {}'.format(cwd))

    # load the model
    sys.path.append(cwd)
    from predictor import ScoringService

    os.chdir('../')

    # change the working directory
    # os.chdir(model_path)
    cwd = os.getcwd()
    print('\nMoved to {}'.format(cwd))

    # load the model
    model_flag = ScoringService.get_model(model_path)

    # models_data = ScoringService.make_model(
    #     waterlevel, 
    #     tidelevel, 
    #     rainfall,
    #     waterlevel_stations
    # )


    in_all_data, out_all_data = make_in_out(start_date, end_date, waterlevel, tidelevel, rainfall, waterlevel_stations)

    print('\nLoading the model...', end = '\r')


    if model_flag:
        print('Loaded the model.   ')
    else:
        print('Could not load the model.')
        return None

    # run all
    predictions = []
    dates = []
    for in_date, in_data in tqdm(in_all_data.items()):
        prediction = ScoringService.predict(in_data)
        if not isinstance(prediction, list):
            print('Invalid data type. Must be list.')
            return None
        dates += [in_date + 1]*len(prediction)
        predictions += prediction

    predictions = pd.DataFrame(predictions)
    columns = set(predictions.columns)
    if columns != {'hour', 'station', 'value'}:
        print('Invalid data name: {},  Excepted name: {}'.format(columns, {'hour', 'station', 'value'}))
        return None

    predictions['date'] = dates
    if predictions.dtypes['value'] != int and predictions.dtypes['value'] != float:
        print('Invalid data type in the prediction. Must be float or int')
        return None
    
    result = pd.merge(predictions, out_all_data, on=('date', 'hour', 'station'))
    result.to_csv('result.csv', index = False, encoding = 'shift-jis')

    # compute RMSE
    print('\nRMSE:', np.sqrt(((result['value_x']-result['value_y'])**2).mean()))

def setting_path(debug = False):

    if debug == False:
        exec_path = 'src'
        model_path = 'model'
        data_dir = 'input/train'
        start_date = 1642
        end_date = 2190

    elif debug == True:
        exec_path = 'src'
        model_path = 'model'
        data_dir = 'input/train'
        start_date = 200
        end_date = 300


    return exec_path, model_path, data_dir, start_date, end_date

def show_dataframe(df, df_name):

    print(f'{df_name}のshape：{df.shape}')
    print(df.head(3))
    print('*'*80)
    print('*'*80)
    print('*'*80)
    print()
    print()
    print()


if __name__ == '__main__':

    debug = False

    exec_path, model_path, data_dir, start_date, end_date = setting_path(debug)

    main(exec_path, model_path, data_dir, start_date, end_date)