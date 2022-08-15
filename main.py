
import scipy
from numpy import linalg as LA
import numpy as np
import scipy.spatial.distance as distance
from scipy.stats import norm
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
import random
from sklearn import datasets#новая строка из документации к graphtools
import graphtools as gt
from scipy import signal#нужно для построения Вейвлета
from scipy.spatial import Voronoi, voronoi_plot_2d #для построения диаграммы Вороного
#для картинок
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import sys #для записи в файл
from scipy.spatial import Delaunay #for tesselation and triangulation
import seaborn as sns #for heatmap
#
import mpmath
#from mpmath import*# почему-то на эту строку выдаются ошибки
#для рисования картинок с кружочками
import matplotlib.patches
import matplotlib.path
from matplotlib.lines import Line2D


import argparse#from Vadims code
import os#from Vadims code
import pandas as pd#from Vadims code
import sympy
from sympy import Symbol
from sympy import *



#вычисление значения  Гауссова ядра в точке х,y
def gaussian_kernel(x,y,mu1,mu2,sigma):
    '''(x,y) --- точка в которой вычисляем значение'''
    difference=np.array([x-mu1,y-mu2])#вектор разности (x,y) и  (mu1,mu2)
    distance=np.linalg.norm(difference)#норма вектора разности
    K_sigma=1/(sigma*math.sqrt(2*math.pi))*norm.pdf(distance,loc=0,scale=sigma)
    return K_sigma




     #return 1/(2*math.pi*sigma**2)*math.exp(-((x-mu1)**2+(y-mu2)**2)/(2*sigma**2))

#процудура нахождения плотности в точке (x,y), рассчитаная как сумма Гауссиан по списку точек Samples
def density(Samples,x,y,sigma):
    '''samples - полученные точки,
     (x,y) - точка, в которой вычисляем плотность;
     sigma --- дисперсия для Гауссиан'''
    d=0#значение плотности
    for i in Samples:
        d+=gaussian_kernel(x,y,i[0],i[1],sigma)
    return d/len(Samples)

#процудура нахождения плотности в точке (x,y), рассчитаная как сумма квадратов обратных расстояний до  точек Samples
def evkl_density(Samples,x,y):
    '''samples - полученные точки,
     (x,y) - точка, в которой вычисляем плотность;
     sigma --- дисперсия для Гауссиан'''
    d=0#значение плотности
    for i in Samples:
        if ((x-i[0])**2+(y-i[1])**2)!=0:
            d += 1 / ((x - i[0]) ** 2 + (y - i[1]) ** 2)
        else:
            print('point is sample point')
    return d

def create_function_grid(Samples,x0,y0,x1,y1,hx,hy,sigma,D,type):
    X = np.arange(x0, x1, hx)  # (старт, финиш, шаг бинаризации)
    Y = np.arange(y0, y1, hy)  # (старт, финиш, шаг бинаризации)
    # определяем 2D-сетку
    X, Y = np.meshgrid(X, Y)
    Dens_values = np.zeros((len(Y), len(X[1])))
    for l in range(len(X[1])):
        for t in range(len(Y)):
            if type=='gaussian_density':
                Dens_values[t, l] = density(Samples, x0 + l * hx, y0 + t * hy, sigma)
            if type=='evkl_density':
                Dens_values[t, l] = evkl_density(Samples, x0 + l * hx, y0 + t * hy)
            if type=='boltzmann_potential_gaussian':
                Dens_values[t, l] =boltzmann_potential(x0 + l * hx, y0 + t * hy, Samples, sigma, D,'gaussian_density')
            if type=='boltzmann_potential_evkl':
                Dens_values[t, l] =boltzmann_potential(x0 + l * hx, y0 + t * hy, Samples, sigma, D,'evkl_density')
    return X,Y,Dens_values


#рисование поверхности по точкам
def plot_surface(X, Y, Z,fig,ax):
# cтроим график
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    linewidth=0, antialiased=False)
    # диапазон по оси Z:
    minz=np.min(Z)
    maxz=np.max(Z)
    ax.set_zlim(minz-1,maxz+1)

    # настройки осей чтобы было красиво:
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter('{x:.02f}')
    # определение цвета
    fig.colorbar(surf, shrink=0.5, aspect=5)

    #  сохраняем картинку
    plt.savefig("3d_surface.png")
    # показываем картинку
    plt.show()

    return 0

#процудура нахождения прямоугольника, внутри которого находятся все samples
def sample_area(Samples):
    '''create grid
    (x0,y0) --- left down corner'''
    #находим левый нижний (x0,y0) и правый верхний (x1,y1) узла графа. (x0,y0) --- левый нижний угол сетки
    x0=min(Samples, key=lambda j:j[0])[0]
    x1=max(Samples, key=lambda j:j[0])[0]
    y0=min(Samples, key=lambda j:j[1])[1]
    y1=max(Samples, key=lambda j:j[1])[1]

    return x0,y0,x1,y1


#создадим вектор плотностей, соответствующий вершинам сетки,
# которые являются ближайшими к  вершинам графа
# (наверно это можно было сделать раньше и оптимальнне, но пока пусть так)
def create_small_density_vector(Samples,sigma):
    DensVector=[]
    for i in Samples:
        DensVector.append(density(Samples,i[0],i[1],sigma))
    return DensVector


#процудура вычисления потенциала в точках графа (независимо от сетки)
def potential_calculation_on_graph(Samples,P,sigma):
    '''P --- psevdo inversre matrix for Laplacian'''
    #вычислим список плотностей в точках графа
    DensVector=create_small_density_vector(Samples,sigma)
    PotentialVector=[]
    k=0
    for i in Samples:
        #вычислим потенциал как P*Dense
        PotentialVector.append(np.dot(P[k],DensVector))
        k+=1

    return PotentialVector

#процедура, возвращающая потенциал по формуле Больцмана U=-Dln p(x,y),
# где p - плотность с параметром sigma
def boltzmann_potential(x,y,Samples,sigma,D,type):
    if type=='gaussian_density':
        u = -D * np.log(density(Samples, x, y, sigma))
    if type=='evkl_density':
        u = -D * np.log(evkl_density(Samples, x, y))
    return u



#процедура создания массива значений функций potential и density для рисования
#функция фозвращает
            #X - список первых координат точек сетки
            #Y- список вторых координат точек сетки
            #DensityValue - список значений логарифма от плотности



def sinc(x,N):
    if x==0:
        s=1
    else:
        s=math.sin(N*x)/(N*x)
    return s

#
def sinc2d(x,y,N):
    f=sinc((np.linalg.norm([x,y])),N)
    return f

#гладкая функция, принимающая значения Values в точках Samples
#Samples - список координат точек
#Values --- список значений
def smoothing_function_sinc(Samples,Values,x,y,N):
    f=0
    for i in range(len(Samples)):
        f+=Values[i]*sinc(x-Samples[i][0],N)*sinc(y-Samples[i][1],N)
    return f

#нарисуем гладкую картинку (без сеток)
def draw_smooth_functiong_general(x0,x1,hx,y0,y1,hy,Samples,Values,N,fig,ax):#
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})#почему-то эта строка не выполняется
    # Диапазоны по оси X и Y:
    X = np.arange(x0, x1, hx) # (старт, финиш, шаг бинаризации)
    Y = np.arange(y0, y1, hy) # (старт, финиш, шаг бинаризации)
    # определяем 2D-сетку
    X, Y = np.meshgrid(X, Y)
    #M=math.floor(n*h/hh)
    #N=math.floor(m*h/hh)
    Z = np.zeros((len(Y),len(X[1])))

    for l in range(len(X[1])):
        for t in range(len(Y)):
            #print('l,t',l,t)
            Z[t,l]=smoothing_function_sinc(Samples,Values,x0+l*hx,y0+t*hy,N)

    #print('z',Z)
    plot_surface(X, Y, Z,fig,ax)
    #ax=sns.heatmap(Z,center=0,cmap='YlGnBu')
    plt.show()

#процудура, рисующая 3д график функции
def draw_function(x0,y0,x1,y1,h,func,fig,ax):
    X = np.arange(x0, x1, h)  # (старт, финиш, шаг бинаризации)
    Y = np.arange(y0, y1, h)  # (старт, финиш, шаг бинаризации)
    # определяем 2D-сетку
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros((len(Y), len(X[1])))

    for l in range(len(X[1])):
        for t in range(len(Y)):
            Z[t, l] = func(X[t,l],Y[t,l])
    plot_surface(X, Y, Z,fig,ax)

#процудура перечисления матриц графа, записывает их  в файл GraphMatrix
def print_graph_matrix(Samples):
    df = Samples
    # Create graph from data. knn - Number of nearest neighbors (including self)
    G = gt.Graph(df, use_pygsp=True, knn=4)  # df - это матрица KxM в которой хранятся первичные вектора.
    #print('j', G)

    original_stdout=sys.stdout
    FileGraphMatrix=open('GraphMatrix','w')
    sys.stdout=FileGraphMatrix
    #with open('GridPointsValue','a') as file:
    print('matrix'+'\n')

    G.A
    print(' Adjacency matrix: binary version' + '\n')
    print(np.around(G.A, decimals=2))
    #print(' Adjacency matrix' + '\n')
    #print(np.around(G.get_adjacency(), decimals=2))
    print(' The weighted degree' + '\n')
    print(np.around(G.dw, decimals=2))
    print(' Adjacency matrix: K' + '\n')
    print(np.around(G.K, decimals=2))
    # вычислим нормализованный лапласиан графа
    G.compute_laplacian()
    print(' Laplacian' + '\n')
    print(np.around(G.L.A, decimals=2))
    # вычислим нормализованный лапласиан графа и псевдооброатную к нему
    G.compute_laplacian('normalized')
    L_K = G.L.A  # матрица - лапласианг графа
    print(' normalized Laplacian'+'\n')
    print( np.around(L_K, decimals=2))
    PsevdoInverseL_K = LA.pinv(L_K)
    print('PsevdoInverse'+'\n')
    print(np.around(PsevdoInverseL_K, decimals=2))
    sys.stdout = original_stdout
    FileGraphMatrix.close()

#процедура рисующая график плотности или потенциала
def plot_density(s, x0, y0, x1, y1, h, sigma, D, type):
    X, Y, DensValues = create_function_grid(s, x0, y0, x1, y1, h, h, sigma, D, type)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #plt.title('Density sigma=%i'%sigma)
    if type=='gaussian_density':
        plt.title('Density, sigma=' + str(sigma))
    if type=='evkl_density':
        plt.title('Density evkl')
    if type=='boltzmann_potential_gaussian':
        plt.title('boltzmann_potential_gaussian, sigma='+str(sigma))
    plot_surface(X, Y, DensValues, fig, ax)

def main():
    #s=np.loadtxt('UMAP_pr.txt')#читаю данные из файла как матрицу
    s = np.loadtxt('Samples2')  # читаю данные из файла как матрицу

    print('my s',s)

    sigma=0.3
    #print('symb_gaussian_kernel(mu1, mu2, sigma)',symb_gaussian_kernel(1, 0, sigma))
    #print('symb_density',symb_density(s,sigma))
    print_graph_matrix(s)

    df=s
    #Create graph from data. knn - Number of nearest neighbors (including self)
    G = gt.Graph(df, use_pygsp=True, knn=4)#df - это матрица KxM в которой хранятся первичные вектора.
    G.A
    #вычислим нормализованный лапласиан графа и псевдооброатную к нему
    G.compute_laplacian('normalized')
    L_K=G.L.A#матрица - лапласианг графа
    #print('Laplacian', L_K)
    PsevdoInverseL_K=LA.pinv(L_K)
    print('P',np.around(PsevdoInverseL_K,decimals=2))

    #нарисуем граф
    G.set_coordinates(kind=s)
    G.plot()
    plt.show()

    #DensV=create_small_density_vector(s,m,n,h,x0,y0,GridPoints)
    P=PsevdoInverseL_K#псевдообратная матрица к лапласиану графа

    #плотность
    x0,y0,x1,y1=sample_area(s)
    h=0.25
    N=1
    #sigma=0.3
    D=1

    sigma_arr=np.array([0.3,0.7,1,2])
    sigma_arr = np.array([0.3, 0.7])
    for sigma in sigma_arr:
        plot_density(s, x0, y0, x1, y1, h, sigma, D, 'gaussian_density')
        plot_density(s, x0, y0, x1, y1, h, sigma, D, 'boltzmann_potential_gaussian')
    plot_density(s, x0, y0, x1, y1, h, sigma, D, 'evkl_density')




    X, Y, DensValues = create_function_grid(s, x0, y0, x1, y1, h, h, sigma, D, 'boltzmann_potential_evkl')
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title('boltzmann_potential_evkl')
    plot_surface(X, Y, DensValues, fig, ax)




    PotentialVector=potential_calculation_on_graph(s, P, sigma)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title('Smooth PotentialLandscape')
    draw_smooth_functiong_general(x0, x1, h, y0, y1, h, s, PotentialVector, N, fig, ax)


    #посомотрим что такое потенциал Больцмана
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #plt.title('Boltzmann')
    def b_p(x,y):
        return boltzmann_potential(x,y,s,sigma,1)
    #draw_function(x0,y0,x0+m*h,y0+n*h,hh,b_p,fig,ax)


    #построение и изображение диаграммы Вороного
    #draw_voronoi_diagramm(s)
    #print('vorVertices',vor.vertices)#вершины диаграммы Вороного
    #триангуляция Делоне
    tri=Delaunay(s)
    #нарисуем результат триангуляции
    plt.title('Delaunay triangulation')
    plt.triplot(s[:,0],s[:,1],tri.simplices)
    plt.plot(s[:,0],s[:,1],'o')
    plt.show()

    #print(tri.simplices)

    #посомотрим что такое sinc2d
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title('Sinc2d')

    def f(x, y):
        return sinc2d(x, y, 3)

    draw_function(-5, -5, 5, 5, h, f,fig,ax)




#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#plt.title('erunda')
#plot_surface(X, Y, Z)
if __name__ == '__main__':
    main()


