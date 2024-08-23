from urllib import request
from django.shortcuts import render
from django.http import HttpResponse
from django.template import RequestContext
from django.shortcuts import redirect

from django.shortcuts import render
import matplotlib.pyplot as plt
import io
import base64
import urllib.parse


from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import Scatter

from .models import Users
import pandas as pd
import numpy as np
import json,os

import yfinance as yf
import datetime as dt
import qrcode

from .models import Project
from django.contrib import messages

from sklearn.linear_model import LinearRegression
from sklearn import preprocessing, model_selection, svm




# The Home page when Server loads up
def index(request):
    user_count = Users.objects.count()

    # ================================================= Left Card Plot =========================================================
    # Here we use yf.download function
 
    data = yf.download(
        
        # passes the ticker
        tickers=['AAPL', 'AMZN', 'QCOM', 'META', 'NVDA', 'JPM'],
        
        group_by = 'ticker',
        
        threads=True, # Set thread value to true
        
        # used for access data[ticker]
        period='1mo', 
        interval='1d'
    
    )

    data.reset_index(level=0, inplace=True)



    fig_left = go.Figure()
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['AAPL']['Adj Close'], name="AAPL")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['AMZN']['Adj Close'], name="AMZN")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['QCOM']['Adj Close'], name="QCOM")
            )
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['META']['Adj Close'], name="META")
            )
  
    fig_left.add_trace(
                go.Scatter(x=data['Date'], y=data['JPM']['Adj Close'], name="JPM")
            )
    fig_left.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")

    plot_div_left = plot(fig_left, auto_open=False, output_type='div')


    # ================================================ To show recent stocks ==============================================
    
    df1 = yf.download(tickers = 'AAPL', period='1d', interval='1d')
    df2 = yf.download(tickers = 'AMZN', period='1d', interval='1d')
    df3 = yf.download(tickers = 'GOOGL', period='1d', interval='1d')
    df4 = yf.download(tickers = 'META', period='1d', interval='1d')

    df6 = yf.download(tickers = 'JPM', period='1d', interval='1d')

    df1.insert(0, "Ticker", "AAPL")
    df2.insert(0, "Ticker", "AMZN")
    df3.insert(0, "Ticker", "QCOM")
    df4.insert(0, "Ticker", "META")

    df6.insert(0, "Ticker", "JPM")

    df = pd.concat([df1, df2, df3, df4,  df6], axis=0)
    df.reset_index(level=0, inplace=True)
    df.columns = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
    convert_dict = {'Date': object}
    df = df.astype(convert_dict)
    df.drop('Date', axis=1, inplace=True)

    json_records = df.reset_index().to_json(orient ='records')
    recent_stocks = []
    recent_stocks = json.loads(json_records)

    # ========================================== Page Render section =====================================================
    return render(request, 'index.html', {
        'plot_div_left': plot_div_left,
        'recent_stocks': recent_stocks,
        "message":"","Users":user_count
    })

def search(request):
    return render(request, 'search.html', {})


   

def ticker(request):
    # ================================================= Load Ticker Table ================================================
    ticker_df = pd.read_csv('app/Data/new_tickers.csv') 
    json_ticker = ticker_df.reset_index().to_json(orient ='records')
    ticker_list = []
    ticker_list = json.loads(json_ticker)


    return render(request, 'ticker.html', {
        'ticker_list': ticker_list
    })

def login(request):
    return render(request, 'login.html',)
#

# The Predict Function to implement Machine Learning as well as Plotting
def predict(request, ticker_value, number_of_days,state=None):
    try:
        # ticker_value = request.POST.get('ticker')
        ticker_value = ticker_value.upper()
        df = yf.download(tickers = ticker_value, period='1d', interval='1m')
        print("Downloaded ticker = {} successfully".format(ticker_value))
    except:
        return render(request, 'API_Down.html', {})

    try:
        # number_of_days = request.POST.get('days')
        number_of_days = int(number_of_days)
    except:
        return render(request, 'Invalid_Days_Format.html', {})
    path=os.path.dirname(os.path.abspath(__file__))+"\data.json"
    with open(path, 'r') as f:
        data=json.load(f)
    Valid_Ticker =data['data']

    # if ticker_value == "" or number_of_days == "":
    #     return render(request, 'Negative_Days.html', {})

    if ticker_value not in Valid_Ticker:
        return render(request, 'Invalid_Ticker.html', {})
    
    if number_of_days <= 0:
        return render(request, 'Negative_Days.html', {"context":"please enter correct number of days"})
    
    if number_of_days > 365:
        return render(request, 'Overflow_days.html', {})
    

    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'], name = 'market data'))
    fig.update_layout(
                        title='{} live share price evolution'.format(ticker_value),
                        yaxis_title='Stock Price (USD per Shares)')
    fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=list([
            dict(count=15, label="15m", step="minute", stepmode="backward"),
            dict(count=45, label="45m", step="minute", stepmode="backward"),
            dict(count=1, label="HTD", step="hour", stepmode="todate"),
            dict(count=3, label="3h", step="hour", stepmode="backward"),
            dict(step="all")
        ])
        )
    )
    fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div = plot(fig, auto_open=False, output_type='div')



    # ========================================== Machine Learning ==========================================
    

    try:
        df_ml = yf.download(tickers = ticker_value, period='3mo', interval='1h')
    except:
        ticker_value = 'A'
        df_ml = yf.download(tickers = ticker_value, period='3mo', interval='1m')

    # Fetching ticker values from Yahoo Finance API 
    # df_ml = df_ml[['Adj Close']]
    # forecast_out = int(number_of_days)
    # df_ml['Prediction'] = df_ml[['Adj Close']].shift(-forecast_out)
    # if np.isnan(df_ml['Prediction']).any():
    #   df_ml['Prediction']= np.nan_to_num(df_ml['Prediction'])  # Replace NaN with 0 or another appropriate va
    #   X = np.array(df_ml.drop(['Prediction'],axis=1, inplace=True))#fillna(0, inplace=True
    #   X = preprocessing.scale(X)
    # X_forecast = X[-forecast_out:]
    # X = X[:-forecast_out]
    # y = np.array(df_ml['Prediction'])
    # y = y[:-forecast_out]
    # X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)
    # # Applying Linear Regression
    # clf = LinearRegression()
    # clf.fit(X_train,y_train)
    # # Prediction Score
    # confidence = clf.score(X_test, y_test)
    # # Predicting for 'n' days stock data
    # forecast_prediction = clf.predict(X_forecast)
    # forecast = forecast_prediction.tolist()

    # Assuming df_ml is your initial DataFrame loaded with data from Yahoo Finance
    df_ml = df_ml[['Adj Close']]
    forecast_out = int(number_of_days)

    # Creating a 'Prediction' column with shifted 'Adj Close' for future price prediction
    df_ml['Prediction'] = df_ml[['Adj Close']].shift(-forecast_out)

    # Replacing NaN in 'Prediction' with 0
    df_ml['Prediction'] = df_ml['Prediction'].fillna(0)
    try:
        # Preparing feature dataset X
        X = np.array(df_ml.drop(['Prediction'], axis=1))

        # Scaling feature dataset
        X = preprocessing.scale(X)
    except:
        return render(request, 'API_Down.html', {})
    # Creating labels dataset y
    y = np.array(df_ml['Prediction'])

    # Splitting data for training and testing, leaving out the last 'forecast_out' days for prediction
    X_forecast = X[-forecast_out:]
    X = X[:-forecast_out]
    y = y[:-forecast_out]

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    # Applying Linear Regression
    clf = LinearRegression()
    clf.fit(X_train, y_train)

    # Prediction score


    confidence = clf.score(X_test, y_test)
    confidence = format(confidence,".4f",)
    # print(f"Model Confidence: {confidence}")

    # Predicting for 'forecast_out' days stock data
    forecast_prediction = clf.predict(X_forecast)
    forecast = forecast_prediction.tolist()
    # print(f"{forecast}")


    # ========================================== Plotting predicted data ======================================

    pred_dict = {"Date": [], "Prediction": []}
    pred_fig = go.Figure()
    for i in range(len(forecast)):
        pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        pred_dict["Prediction"].append(forecast[i])
        # pred_dict["Date"].append(dt.datetime.today() + dt.timedelta(days=i))
        # pred_dict["Prediction"].append(forecast[i])
    
    pred_df = pd.DataFrame(pred_dict)
    # print("the precicted dif is ",pred_df)
    pred_fig = go.Figure([go.Scatter(x=pred_df['Date'], y=pred_df['Prediction'])])
    pred_fig.update_xaxes(rangeslider_visible=True)
    pred_fig.update_layout(paper_bgcolor="#14151b", plot_bgcolor="#14151b", font_color="white")
    plot_div_pred = plot(pred_fig, auto_open=False, output_type='div')

    # ========================================== Display Ticker Info ==========================================

    ticker = pd.read_csv('app/Data/Tickers.csv')
    to_search = ticker_value
    ticker.columns = ['Symbol', 'Name', 'Last_Sale', 'Net_Change', 'Percent_Change', 'Market_Cap',
                    'Country', 'IPO_Year', 'Volume', 'Sector', 'Industry']
    for i in range(0,ticker.shape[0]):
        if ticker.Symbol[i] == to_search:
            Symbol = ticker.Symbol[i]
            Name = ticker.Name[i]
            Last_Sale = ticker.Last_Sale[i]
            Net_Change = ticker.Net_Change[i]
            Percent_Change = ticker.Percent_Change[i]
            Market_Cap = ticker.Market_Cap[i]
            Country = ticker.Country[i]
            IPO_Year = ticker.IPO_Year[i]
            Volume = ticker.Volume[i]
            Sector = ticker.Sector[i]
            Industry = ticker.Industry[i]
            break

    # ========================================== Page Render section ==========================================
    if state == None: 
        return render(request, "result.html", context={ 'plot_div': plot_div, 
                                                    'confidence' : confidence,
                                                    'forecast': forecast,
                                                    'ticker_value':ticker_value,
                                                    'number_of_days':number_of_days,
                                                    'plot_div_pred':plot_div_pred,
                                                    'Symbol':Symbol,
                                                    'Name':Name,
                                                    'Last_Sale':Last_Sale,
                                                    'Net_Change':Net_Change,
                                                    'Percent_Change':Percent_Change,
                                                    'Market_Cap':Market_Cap,
                                                    'Country':Country,
                                                    'IPO_Year':IPO_Year,
                                                    'Volume':Volume,
                                                    'Sector':Sector,
                                                    'Industry':Industry,
                                                    })

    else:
        return {                                                 
                                                    'plot_div_pred':plot_div_pred,
                                                  
                                                    }




def register(request):
    if request.method == 'POST':
        user = request.POST.get('User')
        email = request.POST.get('Email')
        password = request.POST.get('Pass')
        if user != "" and email != "" and password != "":
            new_user = Users(username=user, email=email,password=password)
            
            new_user.save()

            return render(request, "login.html", {"message": "User created successfully!"})
    else:
        return HttpResponse("Invalid request method.", status=405)
def all_data(request):
    ticker_df = pd.read_csv('app/Data/Tickers.csv') 
    json_ticker = ticker_df.reset_index().to_json(orient ='records')
    ticker_list = []
    ticker_list = json.loads(json_ticker)


    return render(request, 'all_data.html', {
        'ticker_list': ticker_list})



def check(request):
    if request.method == 'POST':
        user = request.POST.get('User')
        password = request.POST.get('Pass')
        try:
            user_data = Users.objects.get(username=user)
        except:
            user_data = None
        if user_data == None:
            return render(request,"login.html",{"message":"User is not recoginazed"})
        if user_data.password == password:
            user_count = Users.objects.count()
            request.session["Userka"] = user
            return redirect("index")
        else:
            return render(request,"login.html",{"message":"User or password is incorrect"})



def profile(request):
    user_info = Users.objects.get(username=request.session.get("Userka"))
    user_info = [user_info.username,user_info.email,user_info.password]
    
    return render(request,"profile.html",{"message":"","fill":user_info})
def edit(request, id):
    if request.method == "POST":
        username = request.POST["user"]
        password = request.POST["pass"]
        email = request.POST["gmail"]

        if email.endswith("@gmail.com"):
            user = Users.objects.get(username=id)
            user.delete()
            result =  Users(username=username, email=email,password=password)
            result.save()
            user_count = Users.objects.count()

            return redirect("index")

    else:
        user_details = Users.objects.get(id)
        users = Users.objects.get()
        return render(request, "profile.html", {"fill": user_details, "users": users})

def logout(request):
    return render(request, "login.html",)





def comper(request):
    return render(request, "comper.html",)



def select(request):

    ticker_list = get_data()


    return render(request, 'select.html', {
        'ticker_list': ticker_list
    })
  













# The Predict Function to implement Machine Learning as well as Plotting
def get_data():
    ticker_df = pd.read_csv('app/Data/new_tickers.csv') 
    json_ticker = ticker_df.reset_index().to_json(orient ='records')
    ticker_list = []
    ticker_list = json.loads(json_ticker)
    return ticker_list
def predict_two(request):
    ticker_list=get_data()

    Restult={}
    tickerka = request.POST.getlist("tickerka")
    num_days =int(request.POST.get("days"))
    hasData = False
    if num_days <= 0:
        return render(request, 'Negative_Days.html', {"context":"please enter correct number of days"})
    
    if num_days > 365:
        return render(request, 'Overflow_days.html', {})
    
    for i in tickerka:
        res = predict(request, i,num_days,"Haa")
        # print("the response is 001",num_days)

        if "plot_div_pred" in res:
            Restult[i] = res["plot_div_pred"]
            
            
        else:
            return render(request, 'select.html', {"error":i+" not founded","ticker_list":ticker_list})

    return render(request, 'Res.html', {"data":Restult , "num_days": num_days})
