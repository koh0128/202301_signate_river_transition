import pandas as pd
import time
import pickle
import glob
import numpy as np
import warnings
warnings.simplefilter('ignore')
# from sklearn.ensemble import RandomForestRegressor as lgbm
# import sklearn

class ScoringService(object):
    @classmethod
    def get_model(cls, model_path):

        """Get model method

        Args:
            model_path (str): Path to the trained model directory.

        Returns:
            bool: The return value. True for success.
        """

        #fold1
        cls.model_1_0 = pickle.load(open(model_path + '/lgbm_model_fold1_00.pkl', 'rb'))
        cls.model_1_1 = pickle.load(open(model_path + '/lgbm_model_fold1_01.pkl', 'rb'))
        cls.model_1_2 = pickle.load(open(model_path + '/lgbm_model_fold1_02.pkl', 'rb'))
        cls.model_1_3 = pickle.load(open(model_path + '/lgbm_model_fold1_03.pkl', 'rb'))
        cls.model_1_4 = pickle.load(open(model_path + '/lgbm_model_fold1_04.pkl', 'rb'))
        cls.model_1_5 = pickle.load(open(model_path + '/lgbm_model_fold1_05.pkl', 'rb'))
        cls.model_1_6 = pickle.load(open(model_path + '/lgbm_model_fold1_06.pkl', 'rb'))
        cls.model_1_7 = pickle.load(open(model_path + '/lgbm_model_fold1_07.pkl', 'rb'))
        cls.model_1_8 = pickle.load(open(model_path + '/lgbm_model_fold1_08.pkl', 'rb'))
        cls.model_1_9 = pickle.load(open(model_path + '/lgbm_model_fold1_09.pkl', 'rb'))
        cls.model_1_10 = pickle.load(open(model_path + '/lgbm_model_fold1_10.pkl', 'rb'))
        cls.model_1_11 = pickle.load(open(model_path + '/lgbm_model_fold1_11.pkl', 'rb'))
        cls.model_1_12 = pickle.load(open(model_path + '/lgbm_model_fold1_12.pkl', 'rb'))
        cls.model_1_13 = pickle.load(open(model_path + '/lgbm_model_fold1_13.pkl', 'rb'))
        cls.model_1_14 = pickle.load(open(model_path + '/lgbm_model_fold1_14.pkl', 'rb'))
        cls.model_1_15 = pickle.load(open(model_path + '/lgbm_model_fold1_15.pkl', 'rb'))
        cls.model_1_16 = pickle.load(open(model_path + '/lgbm_model_fold1_16.pkl', 'rb'))
        cls.model_1_17 = pickle.load(open(model_path + '/lgbm_model_fold1_17.pkl', 'rb'))
        cls.model_1_18 = pickle.load(open(model_path + '/lgbm_model_fold1_18.pkl', 'rb'))
        cls.model_1_19 = pickle.load(open(model_path + '/lgbm_model_fold1_19.pkl', 'rb'))
        cls.model_1_20 = pickle.load(open(model_path + '/lgbm_model_fold1_20.pkl', 'rb'))
        cls.model_1_21 = pickle.load(open(model_path + '/lgbm_model_fold1_21.pkl', 'rb'))
        cls.model_1_22 = pickle.load(open(model_path + '/lgbm_model_fold1_22.pkl', 'rb'))
        cls.model_1_23 = pickle.load(open(model_path + '/lgbm_model_fold1_23.pkl', 'rb'))


        #fold2
        cls.model_2_0 = pickle.load(open(model_path + '/lgbm_model_fold2_00.pkl', 'rb'))
        cls.model_2_1 = pickle.load(open(model_path + '/lgbm_model_fold2_01.pkl', 'rb'))
        cls.model_2_2 = pickle.load(open(model_path + '/lgbm_model_fold2_02.pkl', 'rb'))
        cls.model_2_3 = pickle.load(open(model_path + '/lgbm_model_fold2_03.pkl', 'rb'))
        cls.model_2_4 = pickle.load(open(model_path + '/lgbm_model_fold2_04.pkl', 'rb'))
        cls.model_2_5 = pickle.load(open(model_path + '/lgbm_model_fold2_05.pkl', 'rb'))
        cls.model_2_6 = pickle.load(open(model_path + '/lgbm_model_fold2_06.pkl', 'rb'))
        cls.model_2_7 = pickle.load(open(model_path + '/lgbm_model_fold2_07.pkl', 'rb'))
        cls.model_2_8 = pickle.load(open(model_path + '/lgbm_model_fold2_08.pkl', 'rb'))
        cls.model_2_9 = pickle.load(open(model_path + '/lgbm_model_fold2_09.pkl', 'rb'))
        cls.model_2_10 = pickle.load(open(model_path + '/lgbm_model_fold2_10.pkl', 'rb'))
        cls.model_2_11 = pickle.load(open(model_path + '/lgbm_model_fold2_11.pkl', 'rb'))
        cls.model_2_12 = pickle.load(open(model_path + '/lgbm_model_fold2_12.pkl', 'rb'))
        cls.model_2_13 = pickle.load(open(model_path + '/lgbm_model_fold2_13.pkl', 'rb'))
        cls.model_2_14 = pickle.load(open(model_path + '/lgbm_model_fold2_14.pkl', 'rb'))
        cls.model_2_15 = pickle.load(open(model_path + '/lgbm_model_fold2_15.pkl', 'rb'))
        cls.model_2_16 = pickle.load(open(model_path + '/lgbm_model_fold2_16.pkl', 'rb'))
        cls.model_2_17 = pickle.load(open(model_path + '/lgbm_model_fold2_17.pkl', 'rb'))
        cls.model_2_18 = pickle.load(open(model_path + '/lgbm_model_fold2_18.pkl', 'rb'))
        cls.model_2_19 = pickle.load(open(model_path + '/lgbm_model_fold2_19.pkl', 'rb'))
        cls.model_2_20 = pickle.load(open(model_path + '/lgbm_model_fold2_20.pkl', 'rb'))
        cls.model_2_21 = pickle.load(open(model_path + '/lgbm_model_fold2_21.pkl', 'rb'))
        cls.model_2_22 = pickle.load(open(model_path + '/lgbm_model_fold2_22.pkl', 'rb'))
        cls.model_2_23 = pickle.load(open(model_path + '/lgbm_model_fold2_23.pkl', 'rb'))


        #fold3
        cls.model_3_0 = pickle.load(open(model_path + '/lgbm_model_fold3_00.pkl', 'rb'))
        cls.model_3_1 = pickle.load(open(model_path + '/lgbm_model_fold3_01.pkl', 'rb'))
        cls.model_3_2 = pickle.load(open(model_path + '/lgbm_model_fold3_02.pkl', 'rb'))
        cls.model_3_3 = pickle.load(open(model_path + '/lgbm_model_fold3_03.pkl', 'rb'))
        cls.model_3_4 = pickle.load(open(model_path + '/lgbm_model_fold3_04.pkl', 'rb'))
        cls.model_3_5 = pickle.load(open(model_path + '/lgbm_model_fold3_05.pkl', 'rb'))
        cls.model_3_6 = pickle.load(open(model_path + '/lgbm_model_fold3_06.pkl', 'rb'))
        cls.model_3_7 = pickle.load(open(model_path + '/lgbm_model_fold3_07.pkl', 'rb'))
        cls.model_3_8 = pickle.load(open(model_path + '/lgbm_model_fold3_08.pkl', 'rb'))
        cls.model_3_9 = pickle.load(open(model_path + '/lgbm_model_fold3_09.pkl', 'rb'))
        cls.model_3_10 = pickle.load(open(model_path + '/lgbm_model_fold3_10.pkl', 'rb'))
        cls.model_3_11 = pickle.load(open(model_path + '/lgbm_model_fold3_11.pkl', 'rb'))
        cls.model_3_12 = pickle.load(open(model_path + '/lgbm_model_fold3_12.pkl', 'rb'))
        cls.model_3_13 = pickle.load(open(model_path + '/lgbm_model_fold3_13.pkl', 'rb'))
        cls.model_3_14 = pickle.load(open(model_path + '/lgbm_model_fold3_14.pkl', 'rb'))
        cls.model_3_15 = pickle.load(open(model_path + '/lgbm_model_fold3_15.pkl', 'rb'))
        cls.model_3_16 = pickle.load(open(model_path + '/lgbm_model_fold3_16.pkl', 'rb'))
        cls.model_3_17 = pickle.load(open(model_path + '/lgbm_model_fold3_17.pkl', 'rb'))
        cls.model_3_18 = pickle.load(open(model_path + '/lgbm_model_fold3_18.pkl', 'rb'))
        cls.model_3_19 = pickle.load(open(model_path + '/lgbm_model_fold3_19.pkl', 'rb'))
        cls.model_3_20 = pickle.load(open(model_path + '/lgbm_model_fold3_20.pkl', 'rb'))
        cls.model_3_21 = pickle.load(open(model_path + '/lgbm_model_fold3_21.pkl', 'rb'))
        cls.model_3_22 = pickle.load(open(model_path + '/lgbm_model_fold3_22.pkl', 'rb'))
        cls.model_3_23 = pickle.load(open(model_path + '/lgbm_model_fold3_23.pkl', 'rb'))


        #fold4
        cls.model_4_0 = pickle.load(open(model_path + '/lgbm_model_fold4_00.pkl', 'rb'))
        cls.model_4_1 = pickle.load(open(model_path + '/lgbm_model_fold4_01.pkl', 'rb'))
        cls.model_4_2 = pickle.load(open(model_path + '/lgbm_model_fold4_02.pkl', 'rb'))
        cls.model_4_3 = pickle.load(open(model_path + '/lgbm_model_fold4_03.pkl', 'rb'))
        cls.model_4_4 = pickle.load(open(model_path + '/lgbm_model_fold4_04.pkl', 'rb'))
        cls.model_4_5 = pickle.load(open(model_path + '/lgbm_model_fold4_05.pkl', 'rb'))
        cls.model_4_6 = pickle.load(open(model_path + '/lgbm_model_fold4_06.pkl', 'rb'))
        cls.model_4_7 = pickle.load(open(model_path + '/lgbm_model_fold4_07.pkl', 'rb'))
        cls.model_4_8 = pickle.load(open(model_path + '/lgbm_model_fold4_08.pkl', 'rb'))
        cls.model_4_9 = pickle.load(open(model_path + '/lgbm_model_fold4_09.pkl', 'rb'))
        cls.model_4_10 = pickle.load(open(model_path + '/lgbm_model_fold4_10.pkl', 'rb'))
        cls.model_4_11 = pickle.load(open(model_path + '/lgbm_model_fold4_11.pkl', 'rb'))
        cls.model_4_12 = pickle.load(open(model_path + '/lgbm_model_fold4_12.pkl', 'rb'))
        cls.model_4_13 = pickle.load(open(model_path + '/lgbm_model_fold4_13.pkl', 'rb'))
        cls.model_4_14 = pickle.load(open(model_path + '/lgbm_model_fold4_14.pkl', 'rb'))
        cls.model_4_15 = pickle.load(open(model_path + '/lgbm_model_fold4_15.pkl', 'rb'))
        cls.model_4_16 = pickle.load(open(model_path + '/lgbm_model_fold4_16.pkl', 'rb'))
        cls.model_4_17 = pickle.load(open(model_path + '/lgbm_model_fold4_17.pkl', 'rb'))
        cls.model_4_18 = pickle.load(open(model_path + '/lgbm_model_fold4_18.pkl', 'rb'))
        cls.model_4_19 = pickle.load(open(model_path + '/lgbm_model_fold4_19.pkl', 'rb'))
        cls.model_4_20 = pickle.load(open(model_path + '/lgbm_model_fold4_20.pkl', 'rb'))
        cls.model_4_21 = pickle.load(open(model_path + '/lgbm_model_fold4_21.pkl', 'rb'))
        cls.model_4_22 = pickle.load(open(model_path + '/lgbm_model_fold4_22.pkl', 'rb'))
        cls.model_4_23 = pickle.load(open(model_path + '/lgbm_model_fold4_23.pkl', 'rb'))


        #fold5
        cls.model_5_0 = pickle.load(open(model_path + '/lgbm_model_fold5_00.pkl', 'rb'))
        cls.model_5_1 = pickle.load(open(model_path + '/lgbm_model_fold5_01.pkl', 'rb'))
        cls.model_5_2 = pickle.load(open(model_path + '/lgbm_model_fold5_02.pkl', 'rb'))
        cls.model_5_3 = pickle.load(open(model_path + '/lgbm_model_fold5_03.pkl', 'rb'))
        cls.model_5_4 = pickle.load(open(model_path + '/lgbm_model_fold5_04.pkl', 'rb'))
        cls.model_5_5 = pickle.load(open(model_path + '/lgbm_model_fold5_05.pkl', 'rb'))
        cls.model_5_6 = pickle.load(open(model_path + '/lgbm_model_fold5_06.pkl', 'rb'))
        cls.model_5_7 = pickle.load(open(model_path + '/lgbm_model_fold5_07.pkl', 'rb'))
        cls.model_5_8 = pickle.load(open(model_path + '/lgbm_model_fold5_08.pkl', 'rb'))
        cls.model_5_9 = pickle.load(open(model_path + '/lgbm_model_fold5_09.pkl', 'rb'))
        cls.model_5_10 = pickle.load(open(model_path + '/lgbm_model_fold5_10.pkl', 'rb'))
        cls.model_5_11 = pickle.load(open(model_path + '/lgbm_model_fold5_11.pkl', 'rb'))
        cls.model_5_12 = pickle.load(open(model_path + '/lgbm_model_fold5_12.pkl', 'rb'))
        cls.model_5_13 = pickle.load(open(model_path + '/lgbm_model_fold5_13.pkl', 'rb'))
        cls.model_5_14 = pickle.load(open(model_path + '/lgbm_model_fold5_14.pkl', 'rb'))
        cls.model_5_15 = pickle.load(open(model_path + '/lgbm_model_fold5_15.pkl', 'rb'))
        cls.model_5_16 = pickle.load(open(model_path + '/lgbm_model_fold5_16.pkl', 'rb'))
        cls.model_5_17 = pickle.load(open(model_path + '/lgbm_model_fold5_17.pkl', 'rb'))
        cls.model_5_18 = pickle.load(open(model_path + '/lgbm_model_fold5_18.pkl', 'rb'))
        cls.model_5_19 = pickle.load(open(model_path + '/lgbm_model_fold5_19.pkl', 'rb'))
        cls.model_5_20 = pickle.load(open(model_path + '/lgbm_model_fold5_20.pkl', 'rb'))
        cls.model_5_21 = pickle.load(open(model_path + '/lgbm_model_fold5_21.pkl', 'rb'))
        cls.model_5_22 = pickle.load(open(model_path + '/lgbm_model_fold5_22.pkl', 'rb'))
        cls.model_5_23 = pickle.load(open(model_path + '/lgbm_model_fold5_23.pkl', 'rb'))

        cls.station_median = pd.read_csv(model_path + '/station_median.csv')

        return True

    @classmethod
    def make_model(cls, waterlevel, tidelevel, rainfall, waterlevel_stations):

        """make model method

        Args:
            model_path (str): Path to the trained model directory.

        Returns:
            bool: The return value. True for success.
        """

        


        return True


    @classmethod
    def predict(cls, input): # 前日の水位をそのまま予測とするモデル
        """Predict method

        Args:
            input: meta data of the sample you want to make inference from (dict)

        Returns:
            dict: Inference for the given input.

        """

        models_1 = [
            cls.model_1_0,
            cls.model_1_1,
            cls.model_1_2,
            cls.model_1_3,
            cls.model_1_4,
            cls.model_1_5,
            cls.model_1_6,
            cls.model_1_7,
            cls.model_1_8,
            cls.model_1_9,
            cls.model_1_10,
            cls.model_1_11,
            cls.model_1_12,
            cls.model_1_13,
            cls.model_1_14,
            cls.model_1_15,
            cls.model_1_16,
            cls.model_1_17,
            cls.model_1_18,
            cls.model_1_19,
            cls.model_1_20,
            cls.model_1_21,
            cls.model_1_22,
            cls.model_1_23
        ]

        models_2 = [
            cls.model_2_0,
            cls.model_2_1,
            cls.model_2_2,
            cls.model_2_3,
            cls.model_2_4,
            cls.model_2_5,
            cls.model_2_6,
            cls.model_2_7,
            cls.model_2_8,
            cls.model_2_9,
            cls.model_2_10,
            cls.model_2_11,
            cls.model_2_12,
            cls.model_2_13,
            cls.model_2_14,
            cls.model_2_15,
            cls.model_2_16,
            cls.model_2_17,
            cls.model_2_18,
            cls.model_2_19,
            cls.model_2_20,
            cls.model_2_21,
            cls.model_2_22,
            cls.model_2_23
        ]

        models_3 = [
            cls.model_3_0,
            cls.model_3_1,
            cls.model_3_2,
            cls.model_3_3,
            cls.model_3_4,
            cls.model_3_5,
            cls.model_3_6,
            cls.model_3_7,
            cls.model_3_8,
            cls.model_3_9,
            cls.model_3_10,
            cls.model_3_11,
            cls.model_3_12,
            cls.model_3_13,
            cls.model_3_14,
            cls.model_3_15,
            cls.model_3_16,
            cls.model_3_17,
            cls.model_3_18,
            cls.model_3_19,
            cls.model_3_20,
            cls.model_3_21,
            cls.model_3_22,
            cls.model_3_23
        ]

        models_4 = [
            cls.model_4_0,
            cls.model_4_1,
            cls.model_4_2,
            cls.model_4_3,
            cls.model_4_4,
            cls.model_4_5,
            cls.model_4_6,
            cls.model_4_7,
            cls.model_4_8,
            cls.model_4_9,
            cls.model_4_10,
            cls.model_4_11,
            cls.model_4_12,
            cls.model_4_13,
            cls.model_4_14,
            cls.model_4_15,
            cls.model_4_16,
            cls.model_4_17,
            cls.model_4_18,
            cls.model_4_19,
            cls.model_4_20,
            cls.model_4_21,
            cls.model_4_22,
            cls.model_4_23
        ]

        models_5 = [
            cls.model_5_0,
            cls.model_5_1,
            cls.model_5_2,
            cls.model_5_3,
            cls.model_5_4,
            cls.model_5_5,
            cls.model_5_6,
            cls.model_5_7,
            cls.model_5_8,
            cls.model_5_9,
            cls.model_5_10,
            cls.model_5_11,
            cls.model_5_12,
            cls.model_5_13,
            cls.model_5_14,
            cls.model_5_15,
            cls.model_5_16,
            cls.model_5_17,
            cls.model_5_18,
            cls.model_5_19,
            cls.model_5_20,
            cls.model_5_21,
            cls.model_5_22,
            cls.model_5_23
        ]

        stations = input['stations']
        waterlevel = input['waterlevel']
        exception_station_list = ['大谷池', '白川']

        try:

            water_df = pd.DataFrame(waterlevel)
            unique_water_df = water_df[
                ~water_df['station'].isin(exception_station_list)
            ][['station', 'river']].drop_duplicates()

            normal_water_df = water_df[~water_df['station'].isin(exception_station_list)]
            abnormal_water_df = water_df[water_df['station'].isin(exception_station_list)]

            normal_water_df['value'] = normal_water_df['value'].replace({'M':np.nan, '*':np.nan, '-':np.nan, '--':np.nan, '**':np.nan})
            normal_water_df['median'] = normal_water_df.groupby(['station', 'river'])['value'].transform('median')
            normal_water_df['value'] = np.where(normal_water_df['value'].isnull(), normal_water_df['median'], normal_water_df['value'])

            normal_water_df = pd.merge(
                normal_water_df,
                cls.station_median,
                on = 'station',
                how = 'left'
            )
            normal_water_df['value'] = np.where(normal_water_df['value'].isnull(), normal_water_df['station_median'], normal_water_df['value'])

            normal_water_df['value'] = normal_water_df['value'].fillna(0.0)
            normal_water_df['value'] = normal_water_df['value'].astype(float)
            normal_water_df = normal_water_df.drop(columns = ['median', 'station_median'])

            abnormal_water_df['value'] = abnormal_water_df['value'].replace({'M':np.nan, '*':np.nan, '-':np.nan, '--':np.nan, '**':np.nan})
            
            filled_abnormal_water_df = pd.DataFrame()

            for exception_station in exception_station_list:
                
                tmp_abnormal_water_df = abnormal_water_df[abnormal_water_df['station'] == exception_station]

                #1/20追加
                tmp_abnormal_water_df['value'] = tmp_abnormal_water_df['value'].replace({'M':np.nan, '*':np.nan, '-':np.nan, '--':np.nan, '**':np.nan})
    
                tmp_abnormal_water_df['value'] = tmp_abnormal_water_df['value'].fillna(tmp_abnormal_water_df['value'].median())

                #1/20追加
                tmp_abnormal_water_df = pd.merge(
                    tmp_abnormal_water_df,
                    cls.station_median,
                    on = 'station',
                    how = 'left'
                )
                tmp_abnormal_water_df['value'] = np.where(tmp_abnormal_water_df['value'].isnull(), tmp_abnormal_water_df['station_median'], tmp_abnormal_water_df['value'])

                tmp_abnormal_water_df['value'] = tmp_abnormal_water_df['value'].astype(float)
                tmp_abnormal_water_df = tmp_abnormal_water_df.drop(columns = 'station_median')

                filled_abnormal_water_df = pd.concat([filled_abnormal_water_df, tmp_abnormal_water_df], axis = 0).reset_index(drop = True)

            for ii in range(24):

                unique_water_df = pd.merge(
                    unique_water_df,
                    normal_water_df[normal_water_df['hour'] == ii].rename(columns = {'value': f'water_{str(ii).zfill(2)}:00:00'}).drop(columns = 'hour'),
                    on = ['station', 'river'],
                    how = 'left'
                )

            all_pred_water_df = pd.DataFrame()

            for ii in range(24):
                
                model_1 = models_1[ii]
                model_2 = models_2[ii]
                model_3 = models_3[ii]
                model_4 = models_4[ii]
                model_5 = models_5[ii]

                y_pred1 = model_1.predict(unique_water_df.iloc[:, 2:])
                y_pred2 = model_2.predict(unique_water_df.iloc[:, 2:])
                y_pred3 = model_3.predict(unique_water_df.iloc[:, 2:])
                y_pred4 = model_4.predict(unique_water_df.iloc[:, 2:])
                y_pred5 = model_5.predict(unique_water_df.iloc[:, 2:])

                y_pred = (y_pred1 + y_pred2 + y_pred3 + y_pred4 + y_pred5)/5

                tmp_water_df = unique_water_df[['station', 'river']]
                tmp_water_df['hour'] = ii
                tmp_water_df['value'] = y_pred
                all_pred_water_df = pd.concat([all_pred_water_df, tmp_water_df], axis = 0).reset_index(drop = True)

            all_pred_water_df = pd.concat([all_pred_water_df, filled_abnormal_water_df], axis = 0).reset_index(drop = True)

            merged = pd.merge(pd.DataFrame(stations, columns=['station']), all_pred_water_df)
            merged['value'] = merged['value'].replace({'M':0.0, '*':0.0, '-':0.0, '--': 0.0, '**':0.0})
            merged['value'] = merged['value'].fillna(method ='ffill')
            merged['value'] = merged['value'].astype(float)

            prediction = merged[['hour', 'station', 'value']].to_dict('records')
            return prediction

        except:

            merged = pd.merge(pd.DataFrame(stations, columns=['station']), pd.DataFrame(waterlevel))     # 評価対象のみに絞る
            merged['value'] = merged['value'].replace({'M':np.nan, '*':np.nan, '-':np.nan, '--':np.nan, '**':np.nan}) # 欠損値を0.0に入れ替える
            merged['median'] = merged.groupby(['station', 'river'])['value'].transform('median')
            merged['value'] = np.where(merged['value'].isnull(), merged['median'], merged['value'])

            merged = pd.merge(
                merged,
                cls.station_median,
                on = 'station',
                how = 'left'
            )
            merged['value'] = np.where(merged['value'].isnull(), merged['station_median'], merged['value'])

            merged['value'] = merged['value'].fillna(0.0)                                                # その他の欠損値を0.0に入れ替える
            merged['value'] = merged['value'].astype(float)                                              # float型に変換する
            merged = merged.drop(columns = ['median', 'station_median'])

            prediction = merged[['hour', 'station', 'value']].to_dict('records')                         # DataFrameをlist型に変換する

            return prediction
