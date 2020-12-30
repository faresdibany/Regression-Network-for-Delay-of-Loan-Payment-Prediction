# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:56:02 2020

@author: Dell
"""
import numpy as np
import urllib.request as urllib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error  
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import arabic_reshaper
from pandas import read_csv
from sklearn.compose import ColumnTransformer
import math 

# labelEncoder = LabelEncoder()
# oneHotEncoder = OneHotEncoder(categorical_features=[0])
ss = StandardScaler()

dataframe = read_csv("new_set_random.csv",encoding= 'utf8')
# dt = dataframe.astype('string')
# split into input (X) and output (Y) variables
X = dataframe[[ 'النشاط رئيسي', 'النشاط فرعي','الحالة الاجتماعية','عدد الاعالة','النوع','ضامن']].values
# X = dataframe[:,1:10]
Y = dataframe[['الدفع']].values

#Converting the categorical data to numerical for the decision tree
le_sex = LabelEncoder()
le_sex.fit(['متعسر', 'متعسر جدا', 'منتظم', 'مرفوض'])
Y = le_sex.transform(Y) 

le = LabelEncoder()
le.fit(['أرمل','أعزب','أعزب ويعول','متزوج','متزوج ويعول','مطلق','غير متاح'])
X[:,2] = le.transform(X[:,2]) 

le = LabelEncoder()
le.fit(['Male', 'female'])
X[:,4] = le.transform(X[:,4]) 

le = LabelEncoder()
le.fit(['yes', 'no'])
X[:,5] = le.transform(X[:,5]) 

le = LabelEncoder()
le.fit(['القطاع الخدمى','القطاع التجارى','القطاع الزراعى','القطاع الصناعى','لا يوجد نشاط'])
X[:,0] = le.transform(X[:,0]) 

le = LabelEncoder()
le.fit(['اخرى','ادوات كهربائيه','ادوات نظافه','اراضى زراعية','ارانب وطيور','استرجى','اصلاح احذية','اصلاح دراجات و موتوسيكلات','الات زراعية بسيطة','التسمين','الحرفيين','الحلابة','انشطه تجاريه متنوعه','بقاله','بلاستيك','بويات و حدايد','بيع اجهزة و اكسسوار محمول','بيع ادوات صحية','بيع ادوات منزلية','بيع اسماك','بيع اغنام و مواشى','بيع اكسسوار حريمى','بيع غلال وحبوب وعلافة وعطارة','بيع فاكهة و خضار','بيع لحوم - جزارة','بيع مجمدات','بيع مفروشات','بيع منتجات البان','بيع منتجات بترولية','بيع منتجات بلاستيك','بيع منتجات ورقية','بيع موبيليات','تجارة اقمشة و مانيفاتورة','تجارة خرده','تجارة مواد بناء','تجارة و بيع احذيه و منتجات جلديه','تربية اغنام','تربية مواشى','تريية مواشى','تعبئة فحوم - توزيع و بيع بوتاجاز','تعبئة و تجهيز بويات و شحومات','جراج سيارات - محطة بنزين','جلديه','حضانه ومعاهد تعليمية و مكاتب','خدمات متنوعه','خردوات و لوازم خياطيه','خشبيه','دوكو سيارات','رفا و تنظيف و دراى كلين و مكوجى','زجاج','ستوديو تصوير و طبع و تحميض افلام','سجاير و حلويات','سروجى','سمكرى سيارات','سنترالات و خدمات تليفونيه','شحن بطاريات','صيادلة ولوازم صيدليات','صيانة و اصلاح اجهزة كهربائيه','طباعه','غذائية','قطع غيار سيارات و خلافه','كهربائى سيارات','كوافيير و تزيين عرائس و حلاق','مركب صيد','مركبات النقل','مشاتل','مطاعم وقهاوى شعبية','معادن','مقاولات عامه','مكتبة وادوات مكتبية','ملابس جاهزه بيع','ملابس و صناعات نسيجيه','منتجات شمع','مواد بناء','ميكانيكى سيارات','نجارة اخشاب','اصلاح ابواب و شبابيك السيارات', 'لا يوجد نشاط', 'بيع و تأجير شرائط فيديو'])
X[:,1] = le.transform(X[:,1]) 

# X = oneHotEncoder.fit_transform(X).toarray()

# ct = ColumnTransformer([( 'اخرى','ادوات كهربائيه','ادوات نظافه','اراضى زراعية','ارانب وطيور','استرجى','اصلاح احذية','اصلاح دراجات و موتوسيكلات','الات زراعية بسيطة','التسمين','الحرفيين','الحلابة','انشطه تجاريه متنوعه','بقاله','بلاستيك','بويات و حدايد','بيع اجهزة و اكسسوار محمول','بيع ادوات صحية','بيع ادوات منزلية','بيع اسماك','بيع اغنام و مواشى','بيع اكسسوار حريمى','بيع غلال وحبوب وعلافة وعطارة','بيع فاكهة و خضار','بيع لحوم - جزارة','بيع مجمدات','بيع مفروشات','بيع منتجات البان','بيع منتجات بترولية','بيع منتجات بلاستيك','بيع منتجات ورقية','بيع موبيليات','تجارة اقمشة و مانيفاتورة','تجارة خرده','تجارة مواد بناء','تجارة و بيع احذيه و منتجات جلديه','تربية اغنام','تربية مواشى','تريية مواشى','تعبئة فحوم - توزيع و بيع بوتاجاز','تعبئة و تجهيز بويات و شحومات','جراج سيارات - محطة بنزين','جلديه','حضانه ومعاهد تعليمية و مكاتب','خدمات متنوعه','خردوات و لوازم خياطيه','خشبيه','دوكو سيارات','رفا و تنظيف و دراى كلين و مكوجى','زجاج','ستوديو تصوير و طبع و تحميض افلام','سجاير و حلويات','سروجى','سمكرى سيارات','سنترالات و خدمات تليفونيه','شحن بطاريات','صيادلة ولوازم صيدليات','صيانة و اصلاح اجهزة كهربائيه','طباعه','غذائية','قطع غيار سيارات و خلافه','كهربائى سيارات','كوافيير و تزيين عرائس و حلاق','مركب صيد','مركبات النقل','مشاتل','مطاعم وقهاوى شعبية','معادن','مقاولات عامه','مكتبة وادوات مكتبية','ملابس جاهزه بيع','ملابس و صناعات نسيجيه','منتجات شمع','مواد بناء','ميكانيكى سيارات','نجارة اخشاب','اصلاح ابواب و شبابيك السيارات', 'لا يوجد نشاط', 'بيع و تأجير شرائط فيديو' , OneHotEncoder(), [1])], remainder = 'passthrough')
# X[:,1] = ct.fit_transform(X[:,1])

X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, Y, test_size=0.3, random_state=3)
print('Shape of X training set {}'.format(X_trainset.shape),'&',' Size of Y training set {}'.format(y_trainset.shape))
print('Shape of X testing set {}'.format(X_testset.shape),'&',' Size of Y testing set {}'.format(y_testset.shape))

X_trainset = ss.fit_transform(X_trainset)
X_testset = ss.transform(X_testset)
# Neural Network
prediction_network = Sequential()
prediction_network.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=6))
prediction_network.add(Dense(units=10, kernel_initializer='uniform', activation = 'relu'))
prediction_network.add(Dense(units=1, kernel_initializer='uniform'))
prediction_network.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

prediction_network.fit(X_trainset, y_trainset, batch_size=10, epochs=50)

y_pred = prediction_network.predict(X_testset)
rmse = mean_squared_error(y_testset, y_pred)
print("Root Mean Square Error : {:.4f}".format(rmse))

y_pred = np.rint(y_pred)
y_pred = y_pred.astype(int)


le = LabelEncoder()
le.fit(['متعسر', 'متعسر جدا', 'منتظم', 'مرفوض'])
y_pred = le.inverse_transform(y_pred)

le = LabelEncoder()
le.fit(['متعسر', 'متعسر جدا', 'منتظم', 'مرفوض'])
y_testset = le.inverse_transform(y_testset)

