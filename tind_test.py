import pynder
import re
import robobrowser
import itertools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import hstack
import numpy as np
import time
import random


################### SHAMELESSLY STOLEN FROM THE INTERNET #####################################
MOBILE_USER_AGENT = "Mozilla/5.0 (Linux; U; en-gb; KFTHWI Build/JDQ39) AppleWebKit/535.19 (KHTML, like Gecko) Silk/3.16 Safari/535.19"
FB_AUTH = "https://www.facebook.com/v2.6/dialog/oauth?redirect_uri=fb464891386855067%3A%2F%2Fauthorize%2F&display=touch&state=%7B%22challenge%22%3A%22IUUkEUqIGud332lfu%252BMJhxL4Wlc%253D%22%2C%220_auth_logger_id%22%3A%2230F06532-A1B9-4B10-BB28-B29956C71AB1%22%2C%22com.facebook.sdk_client_state%22%3Atrue%2C%223_method%22%3A%22sfvc_auth%22%7D&scope=user_birthday%2Cuser_photos%2Cuser_education_history%2Cemail%2Cuser_relationship_details%2Cuser_friends%2Cuser_work_history%2Cuser_likes&response_type=token%2Csigned_request&default_audience=friends&return_scopes=true&auth_type=rerequest&client_id=464891386855067&ret=login&sdk=ios&logger_id=30F06532-A1B9-4B10-BB28-B29956C71AB1&ext=1470840777&hash=AeZqkIcf-NEW6vBd"
def get_access_token(email, password):
    s = robobrowser.RoboBrowser(user_agent=MOBILE_USER_AGENT, parser="lxml")
    s.open(FB_AUTH)
    ##submit login form##
    f = s.get_form()
    f["pass"] = password
    f["email"] = email
    s.submit_form(f)
    ##click the 'ok' button on the dialog informing you that you have already authenticated with the Tinder app##
    f = s.get_form()
    s.submit_form(f, submit=f.submit_fields['__CONFIRM__'])
    ##get access token from the html response##
    access_token = re.search(r"access_token=([\w\d]+)", s.response.content.decode()).groups()[0]
    #print  s.response.content.decode()
    return access_token

token=get_access_token('email','password')
session= pynder.Session(token)


def swipeall(session):
    girlfriends=session.nearby_users()
    bored=1
    while (bored):
        gf=itertools.islice(girlfriends,30)
        for g in gf:
            try:
                print(g.like())
            except pynder.errors.RequestError:
                print('BOT or ADD')
            sleep_time=random.randint(2,5)
            time.sleep(sleep_time)
            
        x=session.likes_remaining
        if x==1:
            break 

def gather_data(session):
    matches=session.matches()
    user_messages_tup=[]
    for match in itertools.islice(matches,3000):
        user_messages_tup.append((match.user,match.messages ))
    return user_messages_tup


def parse_data(data):
    '''attribute ideas: age,  
    from _data: connection_count, common_like_count, common_likes, common_friends,bio, birth_date, jobs, ''' 
    '''how to measure success: length of conversation'''
    target=[]
    attributes=[]
    for element in data:
        target.append(len(element[1]))

        dt=element[0]._data
        
        #common like
        c_like=element[0]._data['common_like_count']

        #schools
        schools=element[0].schools

        #jobs
        jobs=element[0].jobs

        #bio
        bio=element[0].bio

        #date of birth
        dob=element[0].birth_date

        #put this into target vector
        attributes.append((bio,c_like,schools,jobs,dob))
    return attributes,target

def preprocess(attributes,vect,test):
    '''this method preprocessess all the data. Vect is to be the vectorizer passwed when working with test data. test is a boolean value set 
    to true when dealing with test data'''

    data=attributes

    data_bios=[a[0] if a[0] is not None else '  ' for a in data]

    if not test:
        #process bios
        vectorizer=CountVectorizer(max_features=30,min_df=3)
        parsed_bios=vectorizer.fit_transform(data_bios)
    else:
        #preprocess test data
        parsed_bios=vect.fit(data_bios)

    #common likes
    data_likes=[a[1] for a in data]
    data_likes=np.transpose(np.matrix(data_likes))

    #process schools
    data_schools= [1 if a[2] else 0 for a in data]
    data_schools=np.transpose(np.matrix(data_schools))


    #process jobs
    data_jobs=[1 if a[3] else 0 for a in data]
    data_jobs=np.transpose(np.matrix(data_jobs))

    #process date of birth
    data_dob=[int(str(a[4].year)+str(a[4].month)+str(a[4].day)) for a in data]
    data_dob=np.transpose(np.matrix(data_dob))
    

    #training matrix
    return (hstack((parsed_bios,data_likes,data_schools,data_jobs,data_dob)),vectorizer)


def do_machine_learn(data,target):

    x_train, x_test, y_train, y_test= train_test_split(data,target, test_size=.30)

    reg=LinearRegression().fit(x_train,y_train)
    #print(reg.score(x_test,y_test))


    #lasso regresison is about 56 % on training data
    reg2=Lasso(alpha=.001)
    reg2.fit(x_train,y_train)
    print(reg2.score(x_test,y_test))

    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(x_train,y_train)
    print(knn.score(x_train,y_train))



'''
gd=gather_data(session)
attributes,target=parse_data(gd)
mat,vect= preprocess(attributes,'',0)
do_machine_learn(mat, target)
'''
swipeall(session)

    