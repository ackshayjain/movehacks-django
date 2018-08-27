from django.conf.urls import include, url
from . import views



urlpatterns = [
    url(r'^$',views.index,name='index'),

    
#     url(r'^about/$',views.about,name='about'),
#     url(r'^team/$', views.team, name='team'),
#     url(r'^contact/$',views.contact,name='contact'),
#     url(r'^impact/$',views.impact,name='impact'),
# url(r'^eat/$',views.eat,name='eat'),
#     # url(r'^login/$',views.login,name='login'),
#     # url(r'^register/$',views.register,name='register'),
]
