
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
from scipy.spatial import Voronoi, voronoi_plot_2d #для построения диаграммы Вороного
#для картинок
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import sys #для записи в файл
from scipy.spatial import Delaunay #for tesselation and triangulation
import seaborn as sns #for heatmap

from sympy import *
from VadimsCodeModified import get_rectangles_inside_voronoi,add_to_plot_voronoi_diagram,\
    add_to_plot_rectangles, calc_rect_area_threshold

from potential_old_version import print_graph_matrix

from statistics import median
from create_text_files import print_potential_information
from my_geometry import sample_area,median_distance,sigma_optimal_shi,centroid,create_rectsizes
import pandas as pd

#процедура вычисляющая наименьшее и наибольшее расстояние между точками
def min_max_dist(Samples):
    MinDist = distance.pdist(Samples).min()  # наименьшее расстоние между точками из samples
    MaxDist = distance.pdist(Samples).max()  # наименьшее расстоние между точками из samples
    return MinDist,MaxDist


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

def get_density_vector(Samples,sigma):
    DensVector=[]
    for point in Samples:
        DensVector.append(density(Samples,point[0],point[1],sigma))
    return DensVector


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


#процедура создающая массивы точек (и значений в них), чтобы по ним нарисовать график
#type= определяет значения какой функции брать
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

def sinc(x,N):
    if x==0:
        s=1
    else:
        s=math.sin(N*x)/(N*x)
    return s

#гладкая функция, принимающая значения Values в точках Samples
#Samples - список координат точек
#Values --- список значений
def smoothing_function_sinc(Samples,Values,x,y,N):
    f=0
    for i in range(len(Samples)):
        f+=Values[i]*sinc(x-Samples[i][0],N)*sinc(y-Samples[i][1],N)
    return f

#гладкая функция, принимающая значения Values в точках Samples
#сглаживание в соответствие со значениями массива RectSizes
#Samples - список координат точек
#Values --- список значений
def smoothing_function_size_sinc(Samples,Values,x,y,RectSizes):
    f=0
    for i in range(len(Samples)):
        a=RectSizes[i][0]/2
        b=RectSizes[i][1]/2
        f+=Values[i]*sinc(x-Samples[i][0],math.pi/a)*sinc(y-Samples[i][1],math.pi/b)
    return f


#нарисуем гладкую картинку
def draw_smooth_functiong_general(x0,x1,hx,y0,y1,hy,Samples,Values,N,type_of_smoothing,RectSizes,tit,show):#
    #tit - заглавие картинки
    #show=true - процедура покажет картинку
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    plt.title(tit)
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
            if type_of_smoothing=='homogenous':
                Z[t, l] = smoothing_function_sinc(Samples, Values, x0 + l * hx, y0 + t * hy, N)
            if type_of_smoothing == 'rectangles':
                Z[t, l] = smoothing_function_size_sinc(Samples,Values,x0+l*hx,y0+t*hy,RectSizes)
    plot_surface(X, Y, Z,fig,ax)
    #ax=\
    #fig, axs = plt.subplots(nrows=1, ncols=1)#, figsize=(5, 5), dpi=1000)#new

    #sns.heatmap(Z,center=0,cmap='YlGnBu')

    fig, axs = plt.subplots(nrows=1, ncols=1)#, figsize=(5, 5), dpi=1000)#new
    plt.title(tit)
    axs.pcolormesh(X,Y,Z)
    axs.set_frame_on(False)

    if show:
        plt.show()
    return axs

#процедура рисующая график плотности или потенциала
def plot_density(s, x0, y0, x1, y1, h, sigma, D, type):
    X, Y, DensValues = create_function_grid(s, x0, y0, x1, y1, h, h, sigma, D, type)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #plt.title('Density sigma=%i'%sigma)
    if type=='gaussian_density':
        plt.title('Density, sigma=' + str(np.around(sigma,2)))
    if type=='evkl_density':
        plt.title('Density evkl')
    if type=='boltzmann_potential_gaussian':
        plt.title('boltzmann_potential_gaussian,'+ '\n'+' sigma='+str(np.around(sigma,2)))
    if type == 'boltzmann_potential_evkl':
        plt.title('boltzmann_potential_evkl')
    plot_surface(X, Y, DensValues, fig, ax)
    fig, axs = plt.subplots(nrows=1, ncols=1)  # , figsize=(5, 5), dpi=1000)#new
    plt.title(str(type)+'\n'+' sigma='+str(np.around(sigma,2)))
    axs.pcolormesh(X, Y, DensValues)
    axs.set_frame_on(False)
    plt.show()



def add_to_plot_points(axs, points,clr):
    for p in points:
        axs.scatter(p[0],p[1],color=clr,linewidths=0.2)


def plot_Delaunay(Samples):
    # триангуляция Делоне
    s=Samples
    tri = Delaunay(s)
    # нарисуем результат триангуляции
    plt.title('Delaunay triangulation')
    plt.triplot(s[:, 0], s[:, 1], tri.simplices)
    plt.plot(s[:, 0], s[:, 1], 'o')
    plt.show()


#создадим вектор плотностей  в samples
def create_density_vector(Samples,sigma):
    DensVector=[]
    for i in Samples:
        DensVector.append(density(Samples,i[0],i[1],sigma))
    return DensVector

#процудура вычисления потенциала в точках графа
def potential_calculation_on_graph(Samples,P,sigma):
    '''P --- psevdo inversre matrix for Laplacian'''
    #вычислим список плотностей в точках графа
    DensVector=create_density_vector(Samples,sigma)
    PotentialVector=[]
    # вычислим потенциал как P*Dense
    for k in range(len(Samples)):
        PotentialVector.append(np.dot(P[k],DensVector))
    return PotentialVector

#процедура, возвращающая потенциал по формуле Больцмана U=-Dln p(x,y),
# где p - плотность с параметром sigma
def boltzmann_potential(x,y,Samples,sigma,D,type):
    if type=='gaussian_density':
        u = -D * np.log(density(Samples, x, y, sigma))
    if type=='evkl_density':
        u = -D * np.log(evkl_density(Samples, x, y))
    return u


#процудура, возвращающая список прямоугольников, вписанных только в ограниченные области
def get_rectangles_in_bounded_areas(Samples):
    vor = Voronoi(Samples)
    rectangles = get_rectangles_inside_voronoi(vor)  # прямоугольники вписанные во все области диаграммы
    rectangles1 = []  # прямоугольники вписанные только в ограниченные области диаграммы
    for rect in rectangles:
        if rect != 1:
            rectangles1.append(rect)
    return rectangles1

def get75quartile(rectangles):
    areas=[]
    for rect in rectangles:
        areas.append(rect.area)
    sort_areas=sorted(areas)#сортирует площади по возрастанию
    l=len(areas)
    i=round(3*l/4)
    quart=sort_areas[i]
    return quart

#процедура, возвращающая список центров многоугольников Ваороново, площадь которых больше чем thresh_area
def add_points_to_big_areas(vor,rectangles,thresh_area):
    new_points=[]
    i = 0
    for point_region_idx in vor.point_region:
        coordinates_of_region = \
            np.asarray([vor.vertices[vert_idx] for vert_idx in vor.regions[point_region_idx]])
        # print('coord of region',point_region_idx, coordinates_of_region)
        if rectangles[i] != 1:
            #print('rect area', rectangles[i].area)
            if rectangles[i].area > thresh_area:
                #print('this area is large')
                #print(coordinates_of_region)
                x,y=centroid(coordinates_of_region)
                new_points.append([x,y])

        i += 1
    return new_points

#процедура объединяющая наборы точек и значений
def extended_point_set(s1,s2,v1,v2):
    s=np.concatenate((s1,s2),axis=0)
    v=np.concatenate((v1,v2),axis=0)
    return s,v

def main():
    #s=np.loadtxt('UMAP_pr.txt')#читаю данные из файла как матрицу
    s = np.loadtxt('Samples2')  # читаю данные из файла как матрицу
    #s = np.loadtxt('Samples-test')
    #s = np.loadtxt('Samples-test2')

    print_graph_matrix(s)

    df=s
    #Create graph from data. knn - Number of nearest neighbors (including self)
    G = gt.Graph(df, use_pygsp=True, knn=5)#df - это матрица KxM в которой хранятся первичные вектора.
    G.A
    #вычислим нормализованный лапласиан графа и псевдооброатную к нему
    G.compute_laplacian('normalized')
    L_K=G.L.A#матрица - лапласианг графа
    #print('Laplacian', L_K)
    PsevdoInverseL_K=LA.pinv(L_K)
    #print('P',np.around(PsevdoInverseL_K,decimals=2))

    #нарисуем граф
    G.set_coordinates(kind=s)
    G.plot()
    plt.show()

    #DensV=create_small_density_vector(s,m,n,h,x0,y0,GridPoints)
    P=PsevdoInverseL_K#псевдообратная матрица к лапласиану графа
    # #плотность
    x0,y0,x1,y1=sample_area(s)
    h=0.25 #шаг для построения картинок
    N=1/2 #параметр для сглаживания
    D=1 #параметр диффузии

    #mind, maxd = min_max_dist(s)
    #sigma_arr=np.array([0.3,0.7,1,2])
    #sigma_arr = np.array([0.3, 2])
    #sigma_arr=np.array([2*mind,3*mind,5*mind])

    #for sigma in sigma_arr:
        #plot_density(s, x0, y0, x1, y1, h, sigma, D, 'gaussian_density')
        #plot_density(s, x0, y0, x1, y1, h, sigma, D, 'boltzmann_potential_gaussian')

    sigma=sigma_optimal_shi(s)
    plot_density(s, x0, y0, x1, y1, h, sigma, D, 'gaussian_density')
    #plot_density(s, x0, y0, x1, y1, h, sigma, D, 'boltzmann_potential_gaussian')

    #построение и изображение диаграммы Вороного
    #draw_voronoi_diagramm(s)
    #print('vorVertices',vor.vertices)#вершины диаграммы Вороного
    #plot_Delaunay(s)

    #print(tri.simplices)
    vor = Voronoi(s)
    # кусок из main VadimsCode
    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), dpi=1000)
    add_to_plot_voronoi_diagram(axs, vor)

    rectangles = get_rectangles_inside_voronoi(vor)#прямоугольники вписанные во все области диаграммы
    #print('lenrect', len(rectangles))

    add_to_plot_rectangles(axs, rectangles)

    rectangles1 = get_rectangles_in_bounded_areas(s)  # прямоугольники вписанные только в ограниченные области диаграммы
    print('rect1', rectangles1)
    thresh_area = get75quartile(rectangles1)
    print('quart', thresh_area)
    # quart=calc_rect_area_threshold(rectangles1,part_of_distr=0.75)#part_of_distr=0.75
    # print('quart', quart)
    added_points=add_points_to_big_areas(vor, rectangles, thresh_area)
    add_to_plot_points(axs, added_points,'gold')

    plt.savefig('voronoi_rect.png', format='png')



    print('added points',added_points)

    RectSizes=create_rectsizes(s,rectangles)
    print('rectsize', RectSizes)

    # вычисление потенциала с помошью Лапласиана графа в точках samples и сглаживание
    PotentialVector = potential_calculation_on_graph(s, P, sigma)
    ax=draw_smooth_functiong_general(x0, x1, h, y0, y1, h, s, PotentialVector, N, 'homogenous',
                                  RectSizes,
                                  'Smooth PotentialLandscape,'+ '\n'+' sigma=' + str(np.around(sigma,2))+ '\n'
                                     #+ 'type of smoothing= homogenous',
                                     ,false)
    add_to_plot_points(ax,s,'black')
    add_to_plot_voronoi_diagram(ax, vor)
    add_to_plot_rectangles(ax, rectangles)
    plt.show()

    ax=draw_smooth_functiong_general(x0, x1, h, y0, y1, h, s, PotentialVector, N, 'rectangles', RectSizes,
                                  'Smooth PotentialLandscape, sigma=' + str(
                                   sigma) + 'trype of smoothing= rectangles',false)
    add_to_plot_points(ax, s, 'black')
    add_to_plot_voronoi_diagram(ax, vor)
    add_to_plot_rectangles(ax, rectangles)
    plt.show()


    #азпишем информацию в файл
    DensVector=get_density_vector(s, sigma)
    print_potential_information(s,PotentialVector,DensVector,sigma)

    added_densities=get_density_vector(added_points,sigma)
    added_values =np.zeros(len(added_densities))
    for i in range(len(added_densities)):
        added_values[i]=0.5#-added_densities[i]
    #added_values=np.ones(len(added_points))#положим значение потенциплп=1 во всех новых точках
    extended_s,extended_v=extended_point_set(s,added_points,PotentialVector,added_values)
    ax=draw_smooth_functiong_general(x0,x1,h,y0,y1,h,extended_s,extended_v,N,'homogenous',
                                  RectSizes,
                                  'Extended PotentialLandscape,'+ '\n'+' sigma=' + str(np.around(sigma,2))+ '\n'
                                  #+ 'type of smoothing= homogenous',
                                     ,false)
    add_to_plot_points(ax, s, 'black')
    add_to_plot_voronoi_diagram(ax, vor)
    add_to_plot_rectangles(ax, rectangles)
    add_to_plot_points(ax, added_points, 'red')
    plt.show()




if __name__ == '__main__':
    main()


