from django.contrib import admin
from django.urls import path,include

from app.views import *

urlpatterns = [
    path('',login,name='login'),
    path('check',check,name='check'),
    path('register',register,name='register'),

    path('index',index,name='index'),
    path('profile',profile,name='profile'),

    path('all_data',all_data,name='all_data'),
    path('logout',logout,name='logout'),
    
    path('comper',comper,name='comper'),
    path('select',select,name='select'),
    path('predict_two',predict_two,name='predict_two'),



    
    
    path('search/', search),

    path('edit/<str:id>/',edit,name="edit" ),
    path('predict/<str:ticker_value>/<str:number_of_days>/', predict),
    # path('predict_two/<str:ticker_value>/<str:number_of_days>/', predict_two),
    path('ticker/', ticker),
    
]