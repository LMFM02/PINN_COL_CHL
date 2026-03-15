# **Aplicación de las PINN al control cuantico**

En estos notebooks se hace un barrido de parametros de red, con el objetivo de identificar la configuración de entrenamiento que maximice la fidelidad promedio respecto a resultados previos, tanto para la compuerta Hadamard (1 qubit) y CNOT (2 qubits) --se consideran 3 acoples diferentes--

El barrido se centra en dos hiperparámetros:
- **Tasa de aprendizaje**
- **Número de neuronas por capa**

## **Instalación**

`` # Clonar el repositorio``  
git clone https://github.com/LMFM02/PINN_COL_CHL.git

`` # Acceder a la ruta del repositorio creado``   
cd PINN_COL_CHL

`` # Verificar version de python``  
python3 --version

``# Crear entorno virtual (python ó python3)``  
python3 -m venv sweep_env     
source sweep_env/bin/activate  ``# Linux/Mac``  
sweep_env\Scripts\activate  ``# Windows``   


Para ejecutar los notebooks es necesario tener instaladas las dependencias que se encuentran en el archivo **requirements.txt** en el entorno virtual, puede instalarlas directamente usando: 

``# Instalar dependencias``  
pip install -r requirements.txt

## **Estructura del repositorio**
```  
📁 PINN_COL_CHL/  
├── 📁 Docs/  
|    └── 📁 Referencias_krotov/             #Artículos relacionados con el método de Krotov
│       └── 📄 Effectiveness of the Krotov method in controlling open quantum systems.pdf
|       └── 📄 Optimal control theory for a unitary operation under dissipative evolution.pdf
|       └── 📄 Krotov’s method for quantum optimal control.pdf
├── 📁 Krotov/  
|   └── 📁 src/  
│       └── 📄 objetives.py                 #Metodo fuente para los objetivos, modificada
|   📓 KROTOV_Article_Fanchini_V2.ipynb     #Codigo de la compuerta Hadamard en un qubit
|   📓 KROTOV_CNOT.ipynb                    #Codigo de la compuerta CNOT usando un acople XX
├── 📁 Notebooks/                           #Notebooks de barrido de (lr-n) para la compuerta                          
│   └── 📓 Sweep_CNOT_Hxx.ipynb             #CNOT (acoples Hxx, Hyy, Hzz) y la compuerta Hadamard.
│   └── 📓 Sweep_CNOT_Hyy.ipynb  
|   └── 📓 Sweep_CNOT_Hzz.ipynb  
|   └── 📓 Sweep_Hadamard.ipynb 
├── 📁 src/  
│   └── 📄 gate_pinn_module.py              #Modulo para la prediccion de controles en compuertas
├── 📁 Test/  
│   └── 📓 Best_scenario.ipynb              #Notebook para testear el mejor entrenamiento.
└── 📄 README.md    
├── 📄 requirements.txt                     #Dependencias necesarias para ejecutar los notebooks
```

## **Estructura de los notebooks (PINN)**

En los notebooks se encuentra la estructura necesaria para modelar la compuerta Hadamard y CNOT, de tal forma que sea consistente con el modulo PINN que se encuentra en **src/**.

En cada notebook se planteo la siguiente estructura:
- **Se define el esquema de PINN de acuerdo a la compuerta.** 
- **Se define un conjunto de 500 estados puros aleatorios para evaluar fidelidad promedio.**
- **Se establecen los parametros de red que se asocian al barrido.**

En esta parte de los notebooks se definen todos los parametros de la arquitectura de la PINN 

lr=np.linspace(1e-4,1e-3,10)   
ep=np.arange(5000,30001,5000)    
n=np.arange(100,501,50)   
hl=np.arange(2,10,2)       
chi=np.linspace(1e-5,1e-3,5)    

Estos han sido definidos de forma general para permitir realizar barridos sobre otros hiperparámetros, además de la tasa de aprendizaje y el número de neuronas.


- **Se realiza el barrido de parámetros y evaluación de fidelidad**

Durante el entrenamiento, un bucle itera sucesivamente sobre cada par de hiperparámetros (tasa de aprendizaje y número de neuronas). Para cada combinación, la red predice las funciones de control asociadas a la compuerta cuántica objetivo. Posteriormente, la dinámica controlada predicha es evaluada en QuTiP, donde se calcula la fidelidad promedio utilizando un conjunto de 500 estados cuánticos previamente definidos.

- **Se almacena un archivo binario que contiene todos los entrenamientos.**

Una vez finaliza la ejecución del bucle de entrenamiento, se genera automáticamente un archivo binario que almacena los resultados obtenidos. El nombre del archivo incluye la compuerta considerada, los parámetros que fueron barridos y el rango en el que se realizó. En el caso de la compuerta CNOT, el nombre del archivo también incorpora el tipo de acople utilizado durante el entrenamiento.

Finalmente el archivo binario, contiene los parametros de red del entrenamiento, la fidelidad promedio evaluada, las funciones de control predichas y la perdida tanto total como parcial del modelo de entrenamiento. **El archivo binario generado al finalizar el proceso será el que se compartirá.**

# **Krotov para control cuantico**

Como se muestra en la estructura del repositorio, además de los notebooks de barrido de hiperparámetros con PINN, se incluye una carpeta **Krotov/** con dos notebooks complementarios basados en el método de Krotov para control óptimo cuántico que se encuentra en QuTiP:

- **KROTOV_Article_Fanchini_V2.ipynb** — Desarrollado con el objetivo de reproducir los resultados del artículo del Prof. Felipe Fanchini para la compuerta Hadamard en un sistema de 1 qubit.

- **KROTOV_CNOT.ipynb** — Diseñado como ejemplo de aplicación del método para la compuerta CNOT con 6 funciones de control, considerando un acople transversal de tipo $\sigma_x \otimes \sigma_x$

## **⚠️ ¡Importante!**

El método de Krotov implementado en QuTip admite dos formas de definir la tarea de control cuántico:

- **Objectives** — Permite definir de forma flexible (personalizada) los estados iniciales y objetivos del sistema cuántico.

- **Gate Objectives** — Está orientado exclusivamente a la implementación de compuertas cuánticas. Para esta modalidad, los estados iniciales únicamente pueden ser los estados de la base computacional en la dimensión de la compuerta. Adicionalmente, a través de liouvillian_states, solo se admiten conjuntos de estados específicos predefinidos por el método, entre ellos “full”, “3states”, “d+1”. Ver **Goerz et al. New J. Phys. 16, 055012 (2014) ó Docs/Referencias_krotov/Optimal control theory for a unitary operation under dissipative evolution.**

## **🚨 Advertencia**

- **KROTOV_Article_Fanchini_V2.ipynb** — utiliza la **versión modificada** de `objectives` 
  ubicada en **Krotov/src/**.  Esta implementación reemplaza la clase original de la librería Krotov con el fin de definir los estados (iniciales-objetivos) siguiendo la sección **B. Gate preparation** del artículo **Docs/Referencias_krotov/Effectiveness of the Krotov method in controlling open quantum systems**,  a través del método `gate_objectives()` y el conjunto de estados `fanchini`. Para ejecutar el notebook correctamente, dirijase en el entorno virtual **sweep_env/lib/python3.12/site-packages/krotov/** y en el copie la clase `objectives.py` de la ruta **Krotov/src/**.

- **KROTOV_CNOT.ipynb** —  Este notebook funciona tanto con la clase `objectives` modificada y la original, en este caso como en el ejemplo de ($\sigma_x\otimes\sigma_x$) la tarea cuantica se define mediante transferencias de estados, entonces no hay diferencias con la clase `objectives` original del metodo de krotov, por tanto es equivalente.


