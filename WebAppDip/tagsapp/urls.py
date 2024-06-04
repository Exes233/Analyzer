from django.urls import path, include
from . import views

urlpatterns = [
    path('', views.homepage,name=""),
    path('register', views.register,name="register"),
    path('login', views.login,name="login"),
    path('dashboard', views.dashboard,name="dashboard"),
    path('user-logout', views.user_logout,name="user-logout"),
    path('profile', views.profile,name="profile"),
    path('edit-profile',views.edit_profile,name="edit-profile"),
    path('update-dashboard',views.update_dashboard,name="update-dashboard"),
    path('generate-chart', views.generate_chart, name='generate-chart')
]
