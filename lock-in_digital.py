# -*- coding: utf-8 -*-
"""
Created on Mon May 29 23:19:42 2023

@author: Franco
"""

## LOCK IN AMPLIFIER - DIGITAL
'''
La idea es que con una señal del osciloscopio poder hacerle Fourier y obtener la parte real y la imaginaria
para esto la manera en la que voy a proceder es multiplicar la señal por un seno de la misma frecuencia
y aplicar un filtro pasabajos con una frecuencia de corte 
Luego multiplico la señal por un coseno es decir la misma que antes pero desfasada 90°
y hago lo mismo

PREGUNTA: cual es la diferencia entre integrar en todo el espacio de tiempo y/o pasar la señal por 
un filtro pasabajos?
no cumplen la misma funcion? si yo integro estoy usando ortogonalidad con la funcion que multiplique
y solo me quedo con la amplitud de la componente de la misma frec que la referencia

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.signal import find_peaks
from scipy import signal
import os
os.getcwd()

plt.rc('figure', figsize=(6,5), titlesize=19, facecolor='#EBEBEB')
plt.rc('grid', linestyle='--' )
plt.rc('font', size = 12)
fonts = 16

#%% IMPORTO CSV Y GRAFICO

ruta_carpeta = r'E:\AA-Labo 4\Piezoelectrico\Clase 3\Medicion 1'
M = 0

procesados = pd.read_csv(ruta_carpeta + f'\procesados.csv', index_col = 0)
procesados

R2, esacalas, frecuencias, tita2 = procesados.T.to_numpy()

medicion = pd.read_csv(ruta_carpeta + f'\medicion_{M}.csv', index_col = 0)
t_gen, v_gen, t_piz, v_piz = medicion.T.to_numpy()
frec = frecuencias[M]
medicion

# Grafico la medicion para ver q onda
plt.figure()
plt.plot(t_gen, v_gen,'o--', label = 'Generador')
plt.plot(t_piz, v_piz,'o--', label = 'Piezo')
plt.title(f'Medicion {M}')

plt.legend()
plt.grid()
plt.minorticks_on()
plt.grid(which = 'minor',linestyle=':', linewidth='0.1', color='black' )

refx = np.sin(2*np.pi*frec * t_gen)
refy = np.cos(2*np.pi*frec * t_gen)  

PSDx = v_piz * refx # CHEQUEAR SI HAY Q MULTIPLICAR POR DOS ACA
PSDy = v_piz * refy

#%% PRUEBO CON EL METODO DE INTEGRAR EN EL TIEMPO DE LA MEDICION
def lockin(v,t,f):
    
    dt = t[1]-t[0]
    refx = np.sin(2*np.pi * f * t)
    refy = np.cos(2*np.pi * f * t)  
    
    PSDx = v * refx # CHEQUEAR SI HAY Q MULTIPLICAR POR DOS ACA
    PSDy = v * refy
    
    X = sum(PSDx) * dt * (1/t[-1])
    Y = sum(PSDy) * dt * (1/t[-1])
    
    R = np.sqrt(X**2 + Y**2)
    tita = np.arctan2(Y,X) * 180/(np.pi)
    
    return R, tita

#%%
ruta_carpeta = r'E:\AA-Labo 4\Piezoelectrico\Clase 3\Medicion 1'
procesados = pd.read_csv(ruta_carpeta + f'\procesados.csv', index_col = 0) 
# En ese csv procesados lo unico importante para hacer Fourier es q tengo guardada la frecuencia que use en cada medicion
R2, esacalas, frecuencias, tita2 = procesados.T.to_numpy()

R_piz_arr = []
tita_piz_arr = []
R_gen_arr = []
tita_gen_arr = []

for M in range(340):
    
    medicion = pd.read_csv(ruta_carpeta + f'\medicion_{M}.csv', index_col = 0)
    t_gen, v_gen, t_piz, v_piz = medicion.T.to_numpy()
    frec = frecuencias[M]
    
    R, tita = lockin(v_piz,t_gen,frec)
    R_piz_arr.append(R)
    tita_piz_arr.append(tita)
    
    R, tita = lockin(v_gen,t_gen,frec)
    R_gen_arr.append(R)
    tita_gen_arr.append(tita)

procesados2 = procesados
procesados2['R_gen'] = R_gen_arr
procesados2['R_piz'] = R_piz_arr

procesados2['tita_gen'] = tita_gen_arr
procesados2['tita_piz'] = tita_piz_arr

# procesados2.to_csv(ruta_carpeta + f'\procesados2.csv')
#%% Grafico


plt.figure()
plt.plot(frecuencias, tita_piz_arr,'o--')
plt.grid()
plt.minorticks_on()
plt.grid(which = 'minor',linestyle=':', linewidth='0.1', color='black' )

plt.figure()
plt.semilogy(frecuencias, R_piz_arr,'o--')
plt.grid()
plt.minorticks_on()
plt.grid(which = 'minor',linestyle=':', linewidth='0.1', color='black' )

plt.figure()
plt.plot(frecuencias, R_piz_arr,'o--')
plt.grid()
plt.minorticks_on()
plt.grid(which = 'minor',linestyle=':', linewidth='0.1', color='black' )




#%% Grafico todo junto
ruta_carpeta = r'E:\AA-Labo 4\Piezoelectrico\Clase 3\Medicion 1'

procesados2 = pd.read_csv(ruta_carpeta + f'\procesados2.csv', index_col = 0)
procesados2

raiz2 = np.sqrt(2)
procesados2.columns

maximos = procesados2['maximos']
desfasajes = procesados2['desfasajes']
frecuencias = procesados2['frecuencias']
R = procesados2['R_piz']
tita = procesados2['tita_piz']

ruta_carpeta = 'E:\AA-Labo 4\Piezoelectrico\Clase 3'

datos2 = pd.read_csv(ruta_carpeta + f'\medicion_1.csv', index_col = 0)
datos2

R2, tita2, stdR2, stdtita2, frecuencias2, escalas2 = datos2.T.to_numpy()
tita2 = tita2 % 360 - 180 

f_max = frecuencias2[np.argmax(R2)]
f_min = frecuencias2[np.argmin(R2)]

#Grafico todo
mks = 2

plt.figure()
plt.plot(frecuencias, tita ,'o--', label = 'Osc - Lock-in', ms=mks)
plt.plot(frecuencias, desfasajes,'o--', label = 'Osciloscopio', ms=mks)
plt.plot(frecuencias2, tita2,'o--', label = 'Lock-in', ms=mks)

plt.title('Desfasaje - Escala lineal')
plt.legend()
plt.grid()
plt.minorticks_on()
plt.grid(which = 'minor',linestyle=':', linewidth='0.1', color='black' )

# =============================================================================

lwidth = 2.5
plt.figure()
plt.semilogy(frecuencias2, R2*raiz2,'o--', label = 'Lock-in',  ms = mks,zorder = 0)
#00D688,  AE0094
plt.semilogy(frecuencias, R,'o--',c='#00C27C', label = 'Osc - Lock-in', lw=2,ms=4 , zorder=20)
plt.semilogy(frecuencias, maximos,'o--', label = 'Osciloscopio', ms=mks,zorder = 10)
# plt.axvline(f_max ,ls='--',c='r', label = r'$f_{M}$' + f'={round(f_max)}Hz',ms=4)
plt.axvline(f_min ,ls='--',c='g', label = r'$f_{m}$' + f'={round(f_min)}Hz',ms=4)

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Transferencia')
plt.tight_layout()
# plt.title('Amplitud - Escala logarítmica')

plt.legend()
plt.grid()
plt.minorticks_on()
plt.grid(which = 'minor',linestyle=':', linewidth='0.1', color='black' )


# =============================================================================


plt.figure()
plt.plot(frecuencias, R,'o--', label = 'Osc - Lock-in', ms=mks)
plt.plot(frecuencias, maximos,'o--', label = 'Osciloscopio', ms=mks)
plt.plot(frecuencias2, R2*raiz2,'o--', label = 'Lock-in', ms=mks)

plt.title('Amplitud - Escala lineal')

plt.legend()
plt.grid()
plt.minorticks_on()
plt.grid(which = 'minor',linestyle=':', linewidth='0.1', color='black' )

#%% LO HAGO TAMBIEN PARA EL MODO 2

ruta_carpeta = r'E:\AA-Labo 4\Piezoelectrico\Clase 3\Medicion 3'

procesados2 = pd.read_csv(ruta_carpeta + f'\procesados.csv', index_col = 0)
procesados2

raiz2 = np.sqrt(2)
procesados2.columns

corte = 29
corte_sup = 55
maximos = procesados2['maximos'][corte:-corte_sup]
desfasajes = procesados2['desfasajes'][corte:-corte_sup]
frecuencias = procesados2['frecuencias'][corte:-corte_sup]
# R = procesados2['R_piz']
# tita = procesados2['tita_piz']

#Grafico todo
mks = 2

plt.figure()
# plt.plot(frecuencias, tita*2 ,'o--', label = 'Osc - Lock-in', ms=mks)
plt.plot(frecuencias, desfasajes,'o--', label = 'Osciloscopio', ms=mks)

plt.title('Desfasaje - Escala lineal')

plt.legend()
plt.grid()
plt.minorticks_on()
plt.grid(which = 'minor',linestyle=':', linewidth='0.1', color='black' )

plt.figure()
# plt.semilogy(frecuencias, R,'o--', label = 'Osc - Lock-in', ms=mks)
plt.semilogy(frecuencias, maximos,'o--', label = 'Osciloscopio', ms=mks)

plt.title('Amplitud - Escala logarítmica')

plt.legend()
plt.grid()
plt.minorticks_on()
plt.grid(which = 'minor',linestyle=':', linewidth='0.1', color='black' )

plt.figure()
# plt.plot(frecuencias, R,'o--', label = 'Osc - Lock-in', ms=mks)
plt.plot(frecuencias, maximos,'o--', label = 'Osciloscopio', ms=mks)

plt.title('Amplitud - Escala lineal')

plt.legend()
plt.grid()
plt.minorticks_on()
plt.grid(which = 'minor',linestyle=':', linewidth='0.1', color='black' )

#%% HAGO EL LOCKIN PARA EL MODO 2
ruta_carpeta = r'E:\AA-Labo 4\Piezoelectrico\Clase 3\Medicion 3'
procesados = pd.read_csv(ruta_carpeta + f'\procesados.csv', index_col = 0)


R_piz_arr = []
tita_piz_arr = []
R_gen_arr = []
tita_gen_arr = []

for M in range(corte, 260-corte_sup):
    
    medicion = pd.read_csv(ruta_carpeta + f'\medicion_{M}.csv', index_col = 0)
    t_gen, v_gen, t_piz, v_piz = medicion.T.to_numpy()
    frec = frecuencias[M]
    
    R, tita = lockin(v_piz,t_gen,frec)
    R_piz_arr.append(R)
    tita_piz_arr.append(tita)
    
    R, tita = lockin(v_gen,t_gen,frec)
    R_gen_arr.append(R)
    tita_gen_arr.append(tita)


procesados2 = procesados.iloc[corte:-corte_sup].copy().reset_index(drop=True)

# procesados2[]

procesados2['R_gen'] = R_gen_arr
procesados2['R_piz'] = R_piz_arr

procesados2['tita_gen'] = tita_gen_arr
procesados2['tita_piz'] = tita_piz_arr

# procesados2.to_csv(ruta_carpeta + f'\procesados2.csv')

#%% GRAFICO AL INFORME

ruta_carpeta = r'E:\AA-Labo 4\Piezoelectrico\Clase 3\Medicion 3'

procesados2 = pd.read_csv(ruta_carpeta + f'\procesados2.csv', index_col = 0)
procesados2

raiz2 = np.sqrt(2)
procesados2.columns

maximos = procesados2['maximos']
desfasajes = procesados2['desfasajes']
frecuencias = procesados2['frecuencias']
R = procesados2['R_piz']
tita = procesados2['tita_piz']


f_max = frecuencias[np.argmax(R)]
f_min = frecuencias[np.argmin(R)]
f_max
f_min

mks = 2
vin = 0.98

#Grafico todo

plt.figure()
plt.plot(frecuencias, tita ,'o--', label = 'Osc - Lock-in', ms=mks)
plt.plot(frecuencias, desfasajes,'o--', label = 'Osciloscopio', ms=mks)

plt.title('Desfasaje - Escala lineal')

plt.legend()
plt.grid()
plt.minorticks_on()
plt.grid(which = 'minor',linestyle=':', linewidth='0.1', color='black' )
#-------------------------
plt.figure()
plt.semilogy(frecuencias, R,'o--', label = 'Osc - Lock-in', ms=mks)
plt.semilogy(frecuencias, maximos,'o--', label = 'Osciloscopio', ms=mks)
plt.axvline(f_max ,ls='--',c='r', label = r'$f_{M}$' + f'={round(f_max)}Hz',ms=4)
plt.axvline(f_min ,ls='--',c='g', label = r'$f_{m}$' + f'={round(f_min)}Hz',ms=4)

# plt.title('Amplitud - Escala logarítmica')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Transferencia')
plt.tight_layout()

plt.xticks(rotation = 15, ha = 'right')

plt.legend()
plt.grid()
plt.minorticks_on()
plt.grid(which = 'minor',linestyle=':', linewidth='0.1', color='black' )

#---------------------------------

plt.figure()
plt.plot(frecuencias, R,'o--', label = 'Osc - Lock-in', ms=mks)
plt.plot(frecuencias, maximos,'o--', label = 'Osciloscopio', ms=mks)

plt.title('Amplitud - Escala lineal')

plt.legend()
plt.grid()
plt.minorticks_on()
plt.grid(which = 'minor',linestyle=':', linewidth='0.1', color='black' )





#%% PRUEBO CON EL METODO DEL PASA BAJOS


dt = t_gen[1]-t_gen[0]
t = t_gen
### Defino variables temporales de la medición.
Fs = 1/dt              #[Hz] Frecuencia de muestreo                    
FRef = frec            #[Hz] Frecuencia de la referencia
L = len(t);     #[muestras] Longitud de la señal
# Calculamos variables asociadas 
T = dt;              #[s] Periodo de muestreo
MaxT = L/Fs;           #[s] Tiempo maximo
# t = np.linspace(0,L-1,L)*T;         #[s] Time vector
OmegaRef = FRef*2*np.pi;     #[rad/s] frecuencia angular de referencia
frec
# Elección de tiempo caracteríscito y orden del filtro
fc = 10000          #[Hz] Frecuencia de corte
orden = 6       # Orden del filtro.

#Genero las señales de referencia del lock-in.
Referencia_x = np.array(np.sin(OmegaRef*t));
Referencia_y = np.array(np.cos(OmegaRef*t)); #referencia desfasada en 90º

#%% Demodulación lock-in.

# Generamos una figura
# plt.figure('Amplitud y desfasaje')
plt.figure(1)
fig, ax = plt.subplots(4,1,num=1, sharex=True)
ax[3].set_xlabel('Tiempo [s]')

# 1 Muestro la referencia.
ax[0].plot(t, Referencia_x, label='Referencia')
#ax[0].plot(t, Referencia_y, label='Referencia')
ax[0].set_ylabel('Referencia [V]')
ax[0].grid(True)

# 2 Muestro la señal
ax[1].plot(t, v_piz, label='Señal')
ax[1].set_ylabel('Señal [V]'), ax[1].grid(True)

# 3 Detección de fase. Multiplicación por referencia
PSDx = 2 * v_piz * Referencia_x;
PSDy = 2 * v_piz * Referencia_y; #PSD del segundo canal
# Graficamos señal multiplicada
ax[2].plot(t, PSDx, label='Señal')
# ax[2].plot(t, PSDy, label='Señal')
ax[2].set_ylabel('PSD [V]'), ax[2].grid(True)


# Filtrado de señal 
sos = signal.bessel(orden, 2*np.pi*fc, 'low', fs=Fs,output='sos') # Generación de parametros del filrado
PSDxFiltrada= signal.sosfilt(sos,PSDx) # Filtrado
PSDyFiltrada= signal.sosfilt(sos,PSDy) # Filtrado


## Para graficar R y theta
ax[3].plot(t,np.arctan2(PSDyFiltrada,PSDxFiltrada,), label=r'$\theta$')
ax[3].plot(t,PSDxFiltrada**2+PSDyFiltrada**2,label='R')
## Para graficar X e Y
#ax[3].plot(t,PSDxFiltrada,label='X')
#ax[3].plot(t,PSDyFiltrada,label='Y')
plt.legend(loc='lower right')
ax[3].set_ylabel('Salida'), ax[3].grid(True)


# Ordenamos y salvamos la figura.
plt.tight_layout()  
# plt.savefig(f'seniales.png')
plt.show()


# ESTO NO ME SIRVE PORQUE AL FILTRAR CON EL PASABAJOS ME FILTRA TODA LA SENAL
# XQ ACA NO TENGO UNA SENAL MODULADA QUE QUIERO LEER SINO QUE TENGO UNA SINUSOIDAL PURA
