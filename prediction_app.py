import pickle
import datetime

import numpy as np
import pandas as pd
import catboost as ctb
import streamlit as st

# streamlit run prediction_app.py

class production_prediction():

    def __init__(self):

        self.home_path = ''

        self.fe_operator = pickle.load(open(self.home_path + 'feature_transformation/fe_operator.pkl', 'rb'))
        self.rs_azimuth = pickle.load(open(self.home_path + 'feature_transformation/rs_azimuth.pkl', 'rb'))

        self.mm_md_ft = pickle.load(open(self.home_path + 'feature_transformation/mm_md_ft.pkl', 'rb'))
        self.ss_tvd_ft = pickle.load(open(self.home_path + 'feature_transformation/ss_tvd_ft.pkl', 'rb'))
        self.mm_footage_lateral_length = pickle.load(open(self.home_path + 'feature_transformation/mm_footage_lateral_length.pkl', 'rb'))
        self.mm_p_velocity = pickle.load(open(self.home_path + 'feature_transformation/mm_p-velocity.pkl', 'rb'))
        self.rs_s_velocity = pickle.load(open(self.home_path + 'feature_transformation/rs_s-velocity.pkl', 'rb'))
        self.mm_youngs_modulus = pickle.load(open(self.home_path + 'feature_transformation/mm_youngs_modulus.pkl', 'rb'))
        self.rs_isip = pickle.load(open(self.home_path + 'feature_transformation/rs_isip.pkl', 'rb'))
        self.rs_pump_rate = pickle.load(open(self.home_path + 'feature_transformation/rs_pump_rate.pkl', 'rb'))
        self.mm_total_number_of_stages = pickle.load(open(self.home_path + 'feature_transformation/mm_total_number_of_stages.pkl', 'rb'))
        self.mm_proppant_volume = pickle.load(open(self.home_path + 'feature_transformation/mm_proppant_volume.pkl', 'rb'))
        self.rs_proppant_fluid_ratio = pickle.load(open(self.home_path + 'feature_transformation/rs_proppant_fluid_ratio.pkl', 'rb'))
        self.mm_year_on_production = pickle.load(open(self.home_path + 'feature_transformation/mm_year_on_production.pkl', 'rb'))

        self.mm_production = pickle.load(open(self.home_path + 'feature_transformation/mm_production.pkl', 'rb'))

        self.ctbm_tuned = pickle.load(open(self.home_path + 'model/ctbm_tuned.pkl', 'rb'))

    def data_cleaning(self, df1):

        old_column_names = ['treatment company', 'azimuth', 'md (ft)', 'tvd (ft)', 'date on production', 'operator', 'footage lateral length', 'well spacing', 
                            'porpoise deviation', 'porpoise count', 'shale footage', 'acoustic impedance', 'log permeability', 'porosity', 'poisson ratio', 
                            'water saturation', 'toc', 'vcl', 'p-velocity', 's-velocity', 'youngs modulus', 'isip', 'breakdown pressure', 'pump rate', 
                            'total number of stages', 'proppant volume', 'proppant fluid ratio', 'production']

        new_column_names = [old_name.replace(' ', '_') for old_name in old_column_names]

        df1.columns = new_column_names

        df1['date_on_production'] = pd.to_datetime(df1['date_on_production'])

        return df1
    
    def feature_engineering(self, df2):

        df2['month_on_production'] = df2['date_on_production'].dt.month
        df2['year_on_production'] = df2['date_on_production'].dt.year

        return df2

    def data_preparation(self, df3):

        df3['operator'] = df3['operator'].map(self.fe_operator)
        df3['azimuth'] = self.rs_azimuth.fit_transform(df3[['azimuth']].values)
        df3['md_(ft)'] = self.mm_md_ft.fit_transform(df3[['md_(ft)']].values)
        df3['tvd_(ft)'] = self.ss_tvd_ft.fit_transform(df3[['tvd_(ft)']].values)
        df3['footage_lateral_length'] = self.mm_footage_lateral_length.fit_transform(df3[['footage_lateral_length']].values)
        df3['p-velocity'] = self.mm_p_velocity.fit_transform(df3[['p-velocity']].values)
        df3['s-velocity'] = self.rs_s_velocity.fit_transform(df3[['s-velocity']].values)
        df3['youngs_modulus'] = self.mm_youngs_modulus.fit_transform(df3[['youngs_modulus']].values)
        df3['isip'] = self.rs_isip.fit_transform(df3[['isip']].values)
        df3['pump_rate'] = self.rs_pump_rate.fit_transform(df3[['pump_rate']].values)
        df3['total_number_of_stages'] = self.mm_total_number_of_stages.fit_transform(df3[['total_number_of_stages']].values)
        df3['proppant_volume'] = self.mm_proppant_volume.fit_transform(df3[['proppant_volume']].values)
        df3['proppant_fluid_ratio'] = self.rs_proppant_fluid_ratio.fit_transform(df3[['proppant_fluid_ratio']].values)
        df3['year_on_production'] = self.mm_year_on_production.fit_transform(df3[['year_on_production']].values)

        columns_selected = ['md_(ft)', 'tvd_(ft)', 'operator', 'p-velocity', 's-velocity', 'youngs_modulus', 'isip', 
                            'pump_rate', 'total_number_of_stages', 'proppant_volume', 'proppant_fluid_ratio']

        return df3[columns_selected]

    def get_prediction(self, df4):
        
        df4['production'] = self.mm_production.inverse_transform([self.ctbm_tuned.predict(df4)])

        return df4

predictor = production_prediction()

st.set_page_config(layout='wide')

st.title('Oil Production Prediction App')

st.header('Fill the boxes with data and click on the button below to make predictions')

c1, c2, c3, c4, c5 = st.columns((1, 1, 1, 1, 1))

df = pd.read_csv('data/dataset.csv')

old_column_names = ['treatment company', 'azimuth', 'md (ft)', 'tvd (ft)', 'date on production', 'operator', 'footage lateral length', 'well spacing', 
                    'porpoise deviation', 'porpoise count', 'shale footage', 'acoustic impedance', 'log permeability', 'porosity', 'poisson ratio', 
                    'water saturation', 'toc', 'vcl', 'p-velocity', 's-velocity', 'youngs modulus', 'isip', 'breakdown pressure', 'pump rate', 
                    'total number of stages', 'proppant volume', 'proppant fluid ratio', 'production']

new_column_names = [old_name.replace(' ', '_') for old_name in old_column_names]

df.columns = new_column_names

date_on_production = c1.date_input( "First Production Date:", datetime.date(2022, 1, 1))
c1.text('Selected: {}'.format(date_on_production))

treatment_company = c2.selectbox('Treatement Company:', (df['treatment_company'].unique()))
c2.text('Selected: {}'.format(treatment_company))

operator = c3.selectbox('Operator:', (df['operator'].unique()))
c3.text('Selected: {}'.format(operator))

azimuth = c4.number_input('Well Drilling Direction:')
c4.text('Current Number: {}'.format(azimuth))

md_ft = c5.number_input('Measure Depth (ft):')
c5.text('Current Number: {}'.format(md_ft))

tvd_ft = c1.number_input('True Vertical Depth (ft):')
c1.text('Current Number: {}'.format(tvd_ft))

footage_lateral_length = c2.number_input('Horizontal Well Section:')
c2.text('Current Number: {}'.format(footage_lateral_length))

well_spacing = c3.number_input('Distance to the Closest Nearby Well:')
c3.text('Current Number: {}'.format(well_spacing))

porpoise_deviation = c4.number_input('Porpoise Deviation (ft):')
c4.text('Current Number: {}'.format(porpoise_deviation))

porpoise_count = c5.number_input('Porpoise Count:')
c5.text('Current Number: {}'.format(porpoise_count))

shale_footage = c1.number_input('Shale Footage:')
c1.text('Current Number: {}'.format(shale_footage))

acoustic_impedance = c2.number_input('Acoustic Impedance (ft/s * g/cc):')
c2.text('Current Number: {}'.format(acoustic_impedance))

log_permeability = c3.number_input('Log Permeability:')
c3.text('Current Number: {}'.format(log_permeability))

porosity = c4.number_input('Porosity:')
c4.text('Current Number: {}'.format(porosity))

poisson_ratio = c5.number_input('Poisson Ratio:')
c5.text('Current Number: {}'.format(poisson_ratio))

water_saturation = c1.number_input('Water Saturation:')
c1.text('Current Number: {}'.format(water_saturation))

toc = c2.number_input('Total Organic Carbon:')
c2.text('Current Number: {}'.format(toc))

vcl = c3.number_input('Amount of Clay Minerals:')
c3.text('Current Number: {}'.format(vcl))

p_velocity = c4.number_input('Velocity of P-waves (ft/s):')
c4.text('Current Number: {}'.format(p_velocity))

s_velocity = c5.number_input('Velocity of S-waves (ft/s):')
c5.text('Current Number: {}'.format(s_velocity))

youngs_modulus = c1.number_input('Youngs Modulus (giga pascals):')
c1.text('Current Number: {}'.format(youngs_modulus))

isip = c2.number_input('Instantaneous Shut-in Pressure:')
c2.text('Current Number: {}'.format(isip))

breakdown_pressure = c3.number_input('Breakdown Pressure:')
c3.text('Current Number: {}'.format(breakdown_pressure))

pump_rate = c4.number_input('Pump Rate:')
c4.text('Current Number: {}'.format(pump_rate))

total_number_of_stages = c5.number_input('Total Number of Stages:')
c5.text('Current Number: {}'.format(total_number_of_stages))

proppant_volume = c1.number_input('Proppant Volume (lbs):')
c1.text('Current Number: {}'.format(proppant_volume))

proppant_fluid_ratio = c2.number_input('Proppant Fluid Ratio (lbs/gallon):')
c2.text('Current Number: {}'.format(proppant_fluid_ratio))

if st.button('Predict'):

    production = 0

    df1 = pd.DataFrame({'treatment_company': [treatment_company], 'azimuth': [azimuth], 'md_(ft)': [md_ft], 'tvd_(ft)': [tvd_ft], 
                        'date_on_production': [date_on_production], 'operator': [operator], 'footage_lateral_length': [footage_lateral_length], 
                        'well_spacing': [well_spacing], 'porpoise_deviation': [porpoise_deviation], 'porpoise_count': [porpoise_count], 
                        'shale_footage': [shale_footage], 'acoustic_impedance': [acoustic_impedance], 'log_permeability': [log_permeability], 
                        'porosity': [porosity], 'poisson_ratio': [poisson_ratio], 'water_saturation': [water_saturation], 'toc': [toc], 'vcl': [vcl], 
                        'p-velocity': [p_velocity], 's-velocity': [s_velocity], 'youngs_modulus': [youngs_modulus], 'isip': [isip], 
                        'breakdown_pressure': [breakdown_pressure], 'pump_rate': [pump_rate], 'total_number_of_stages': [total_number_of_stages], 
                        'proppant_volume': [proppant_volume], 'proppant_fluid_ratio': [proppant_fluid_ratio], 'production': [production]})

    df2 = predictor.data_cleaning(df1)
    df3 = predictor.feature_engineering(df2)
    df4 = predictor.data_preparation(df3)
    prediction = predictor.get_prediction(df4)

    production = prediction['production'][0]

    st.header('12 months cumulative production for the well is: {} mmcf'.format(round(production, 2)))

else:
    st.header('Click the button to make a prediction')

# if __name__ == "__main__":