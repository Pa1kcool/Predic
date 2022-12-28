#!/usr/bin/env python
from flask import render_template, flash, request
import logging, io, base64, os, datetime
from datetime import datetime
from datetime import timedelta
import numpy as np  
import pandas as pd  
from sklearn import preprocessing;
from sklearn import model_selection;
from sklearn import linear_model;
from Webapp import app


# global variables
earthquake_live = None
days_out_to_predict = 7


from flask import Flask
app=Flask(__name__)
@app.route('/')

def prepare_earthquake_data_and_model(days_out_to_predict = 7, eta=0.1):
    '''
    Desccription : From extraction to model preparation. This function takes in how many days to predict or rolling window
                    period and learning rate. We extract data directly from https://earthquake.usgs.gov/
                    instead of loading from existing database since we want real time data that is updated every minute.
    
    Arguments : int (days_to_predict rolling window), float (learning rate of alogrithm)

    Return : Pandas Dataframe (Prediction dataframe with live/ future NaN values in outcome magnitutde of quake that has to be predicted)
    '''
    # get latest data from USGS servers
    df = pd.read_csv('https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv')
    df = df.sort_values('time', ascending=True)
    # truncate time from datetime
    df['date'] = df['time'].str[0:10]

    # only keep the columns needed
    df = df[['date', 'latitude', 'longitude', 'depth', 'mag', 'place']]
    temp_df = df['place'].str.split(', ', expand=True) 
    df['place'] = temp_df[1]
    df = df[['date', 'latitude', 'longitude', 'depth', 'mag', 'place']]

    # calculate mean lat lon for simplified locations
    df_coords = df[['place', 'latitude', 'longitude']]
    df_coords = df_coords.groupby(['place'], as_index=False).mean()
    df_coords = df_coords[['place', 'latitude', 'longitude']]

    df = df[['date', 'depth', 'mag', 'place']]
    df = pd.merge(left=df, right=df_coords, how='inner', on=['place'])

    # loop through each zone and apply MA
    eq_data = []
    eq_data_last_days_out = []  
    for symbol in list(set(df['place'])):
        temp_df = eq_tmp[eq_tmp['place'] == place].copy()

        temp_df['Mdep_21'] = temp_df['depth'].rolling(window=21,center=False).mean() 
        temp_df['Mdep_14'] = temp_df['depth'].rolling(window=14,center=False).mean()
        temp_df['Mdep_7'] = temp_df['depth'].rolling(window=7,center=False).mean()
        temp_df['Mmag_21'] = temp_df['mag'].rolling(window=21,center=False).mean() 
        temp_df['Mmag_14'] = temp_df['mag'].rolling(window=14,center=False).mean()
        temp_df['Mmag_7'] = temp_df['mag'].rolling(window=7,center=False).mean()
        temp_df.loc[:, 'mag_outcome'] = temp_df.loc[:, 'Mmag_7'].shift(-9)##donot predict last 8 days as rolling window might be wrong for 28-30
    
    #days to predict value on earth quake data this is not yet seen or witnessed by next 7 days (consider as live next 7 days period)
    
    eq_data_last_days_out.append(temp_df.tail(9))#days we didnot count so we will predict these days

    eq_data.append(temp_df)

    # concat all location-based dataframes into master dataframe
    eq_all = pd.concat(eq_data)
    eq_all#convert list to dataframe

    # remove any NaN fields
    df = df[np.isfinite(df['Mdep21'])]
    df = df[np.isfinite(df['Mmag21'])]
    df = df[np.isfinite(df['mag_outcome'])]
    
    eq_data_last_days_out = pd.concat(eq_data_last_days_out)
    eq_data_last_days_out = eq_data_last_days_out[np.isfinite(eq_data_last_days_out['Mmag_21'])]
    # prepare outcome variable
    #considered magnitude above 2.5 as dangerous hence prediction outcome as '1' elso '0'.
    eq_all['mag_outcome'] = np.where(eq_all['mag_outcome'] > 2.5, 1,0)


    df = df[['date',
             'latitude',
             'longitude',
             'Mdep21',
             'Mdep14',
             'Mdep7',
             'Mmag_21', 
             'Mmag_14',
             'Mmag_7',
             'mag_outcome']]

    # keep only data where we can make predictions
    df_features=eq_all
    # splitting traing and testing dataset with trainging size = 70% and test = 30%
    req=['depth','Mdep_21','Mdep_14','Mdep_7','Mmag_21','Mmag_14','Mmag_7']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df_features[req], df_features['mag_outcome'], test_size=0.3, random_state=42)
    from keras import optimizers
    from keras.utils import plot_model
    from keras.models import Sequential, Model
    from keras.regularizers import L1L2
    from keras.layers.convolutional import Conv1D, MaxPooling1D
    from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Flatten,BatchNormalization,Dropout
    model_mlp = Sequential()
    model_mlp.add(Dense(80, activation='relu', kernel_regularizer=L1L2(l1=1e-4, l2=1e-2), input_dim=X_train.shape[1]))
    model_mlp.add(Dropout(0.2))
    model_mlp.add(Dense(20, activation='relu', kernel_regularizer=L1L2(l1=1e-2, l2=1e-1)))
    model_mlp.add(Dense(1, activation='sigmoid'))
    model_mlp.compile(optimizer='adam',loss='binary_crossentropy',metrics=['AUC'])
    #model_mlp.summary()
    mlp_history = model_mlp.fit(X_train.values, y_train, validation_data=(X_test.values, y_test), epochs=20, verbose=1)
    #10 epochs is good
    #15 is questionable
    model_mlp.evaluate(X_test, y_test)
    #y_test.shape

    from sklearn.metrics import explained_variance_score, mean_poisson_deviance, mean_gamma_deviance
    from sklearn.metrics import r2_score
    from sklearn.metrics import max_error

    # predict probabilities for test set
    yhat_probs = model_mlp.predict(X_test, verbose=0)
    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]

    var = explained_variance_score(y_test.values.reshape(-1,1), yhat_probs)
    #print('Variance: %f' % var)

    r2 = r2_score(y_test.values.reshape(-1,1), yhat_probs)
    #print('R2 Score: %f' % var)

    #plotting the training and validation loss
    #plt.plot(mlp_history.history['loss'], label='train loss')
    #plt.plot(mlp_history.history['val_loss'], label='val loss')
    #plt.xlabel("epoch")
    #plt.ylabel("Loss")
    #plt.legend()
    
    from sklearn import metrics
    predicted = model_mlp.predict(X_test)
    # Evaluating the model
    #print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predicted)))
    #print('R-squared :', metrics.r2_score(y_test, predicted))

#taken from net
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import confusion_matrix

    pred= predicted

    #print(roc_auc_score(y_test, pred))

    fpr, tpr, _ = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)
    #print('AUC:', np.round(roc_auc,4))

    #plt.title('Receiver Operating Characteristic')
    #plt.legend(loc = 'lower right')
    #plt.plot([0, 1], [0, 1],'r--')
    #plt.xlim([0, 1])
    #plt.ylim([0, 1])
    #plt.ylabel('True Positive Rate')
    #plt.xlabel('False Positive Rate')
    #plt.show()
    #predicted

    df_live = pd.concat(df_live)
    df_live = df_live[np.isfinite(df_live['Mmag21'])]
    # add pred to live data
    df_live = df_live[['date', 'place', 'latitude', 'longitude']]
    # add predictions back to dataset 
    df_live = df_live.assign(pred=pd.Series(pred).values)

    # aggregate down dups
    df_live = df_live.groupby(['date', 'place'], as_index=False).mean()

    # increment date to include DAYS_OUT_TO_PREDICT
    df_live['date']= pd.to_datetime(df_live['date'],format='%Y-%m-%d') 
    df_live['date'] = df_live['date'] + pd.to_timedelta(days_out_to_predict,unit='d')

    return(df_live)

def get_earth_quake_estimates(desired_date, df_live):
    '''
    Description : gets desired date to predict earthquake and live prediction dataframe with NaN values as outcome magnitude 
                  probablity that has to be predicted. The function also deals with converting to google maps api format 
                  of location co-ordinates to mark it on the map.

    Arguments : DateTime object (desired_date to predict), Pandas DataFrame (dataframe of prediction with NaN values as outcome)

    Return : string (Google maps api format location coordinates)

    '''
    from datetime import datetime
    live_set_tmp = df_live[df_live['date'] == desired_date]

    # format lat/lons like Google Maps expects
    LatLngString = ''
    if (len(live_set_tmp) > 0):
        for lat, lon, pred in zip(live_set_tmp['latitude'], live_set_tmp['longitude'], live_set_tmp['pred']): 
            # this is the threashold of probability to decide what to show and what not to show
            if (pred > 0.3):
                LatLngString += "new google.maps.LatLng(" + str(lat) + "," + str(lon) + "),"

    return(LatLngString)


@app.before_first_request
def startup():
    global earthquake_live

    # prepare earthquake data, model and get live data set with earthquake forecasts
    earthquake_live = prepare_earthquake_data_and_model()


@app.route("/", methods=['POST', 'GET'])
def build_page():
        if request.method == 'POST':

            horizon_int = int(request.form.get('slider_date_horizon'))
            horizon_date = datetime.today() + timedelta(days=horizon_int)

            return render_template('index.html',
                date_horizon = horizon_date.strftime('%m/%d/%Y'),
                earthquake_horizon = get_earth_quake_estimates(str(horizon_date)[:10], earthquake_live),
                current_value=horizon_int, 
                days_out_to_predict=days_out_to_predict)

        else:
            # set blank map
            return render_template('index.html',
                date_horizon = datetime.today().strftime('%m/%d/%Y'),
                earthquake_horizon = '',
                current_value=0,
                days_out_to_predict=days_out_to_predict)

app.run(host="0.0.0.0", port=5010)


