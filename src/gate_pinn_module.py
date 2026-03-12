"""
Módulo que facilita el planteamiento de las ecuaciones de un sistema físico 
que se rige por la ecuación maestra de Linblad de dimensión arbitraria
y resuelve la evolución temporal del control y la dinámica del sistema
mediante redes neuronales PINN

Python +=3.7
"""
from pathlib import Path
import json
from datetime import date
import sympy
from sympy.physics.quantum.dagger import Dagger as sympy_dag
import qutip
import numpy as np
import matplotlib.pyplot as plt
import copy
import torch #Marco de trabajo para redes neuronales
from torch import nn #Herramientas para elaborar el modelo de red neuronal
from torch.autograd import grad
from torch.utils.data import TensorDataset, DataLoader
from IPython.display import clear_output

#Definimos la clase Sin() que hereda de torch.nn.Module
class Sin(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, input):
        return torch.sin(input)

#En pytorch, todos los esqueletos de redes neuronales se definen como una clase que heredan los métodos y propiedades de torch.nn.Module
class NeuralNetwork(nn.Module):
    def __init__(self, out_components, hid_layers, neurons): 
        super().__init__()
        self.out_components = out_components #Número de componentes que solucionaremos (incluye componentes de matriz densidad y funciones de control)
        self.hid = hid_layers
        self.neurons = neurons
        
        # Lista para almacenar las capas ocultas
        capas_ocultas = []

        # Crea capas ocultas y agrega a la lista
        for _ in range(self.hid-1):
            capa_oculta = nn.Linear(self.neurons, self.neurons)
            activacion = Sin()
            capas_ocultas.extend([capa_oculta, activacion])

        #A partir de aquí se definen las capas de mi red neuronal de forma ordenada:
        self.linear_sin_layers = nn.Sequential(
            nn.Linear(1, self.neurons),
            Sin(),
            *capas_ocultas,
            nn.Linear(self.neurons, self.out_components), #Capa de salida de las componentes de la matriz densidad + campos de control
        )  

    def forward(self, t, condition_keys, control_keys, init_condition_len):
        # El método forward se ejecuta cuando instanciamos NeuralNetwork. Es llamado por __call__() internamente
        # Éste método propaga la matriz/vector de entrada de la red neuronal capa por capa
        
        outputs = self.linear_sin_layers(t)
        sol_outs = {} 
            
        for id in range(init_condition_len):
            #id es el índice que enumera cada lote de soluciones por cada estado inicial. El id=0, es el lote 0 correspondiente al estado inicial 0
            sol_outs[id] = {key: outputs[:,len(condition_keys)*id+i].reshape(-1,1) for i, key in enumerate(condition_keys)} #Almaceno las predicciones de la red por lotes enumerados por id
        
        sol_outs[init_condition_len] = {key: outputs[:,len(condition_keys)*init_condition_len+i].reshape(-1,1) for i, key in enumerate(control_keys)} #El último lote es el lote de las predicciones del control/controles
        #usando el índice self.init_condition_len se puede acceder a las predicciones del control            
        return sol_outs
        
class PhysicsEquations:
    def __init__(self, hamiltonian, L_operators=None, gammas=None):
        self.hamiltonian = hamiltonian
        self.dimension = hamiltonian.shape[0]
        self.L_operators = L_operators
        self.gammas = gammas 
        self.rho_sy = self.density_matrix_create(self.dimension)
        self.disp = self.disipation_terms_create()  
        self.dot_rho = self.master_equation_create()
        self.model_eqn = self.equations_generator()
        self.trace_rho = self.density_matrix_trace()

    @staticmethod
    def density_matrix_create(dimension):
        #Genero la matriz densidad de dimensión arbitraria (la diagonal es real)
        z = []
        for i in range(dimension):
            fila = []
            for j in range(dimension):
                if i == j:
                    fila.append(sympy.symbols(f'rho["rho_{i}{j}"]', real=True))  
                elif j > i:
                    fila.append(sympy.symbols(f'rho["rho_{i}{j}"]', real=True)+sympy.I*sympy.symbols(f'rho["rho_{i}{j}i"]', real=True))
                elif j < i:
                    fila.append(sympy.symbols(f'rho["rho_{j}{i}"]', real=True)-sympy.I*sympy.symbols(f'rho["rho_{j}{i}i"]', real=True))
            z.append(fila)
            
        return sympy.Matrix(z)

    def disipation_terms_create(self):
        #Genero los términos asociados a la disipación
        disp = sympy.zeros(self.dimension, self.dimension)
        if self.L_operators is None and self.gammas is None:
            return disp
        else:
            for L_i, gamma_i in zip(self.L_operators, self.gammas):
                disp += gamma_i*L_i
            return disp

    def master_equation_create(self):
        #Genero Ecuación maestra en formato matricial
        return -sympy.I*(self.hamiltonian*self.rho_sy - self.rho_sy*self.hamiltonian) + self.disp

    def equations_generator(self):
        # Genero todas las ecuaciones de pérdida física que se deben introducir en la red neuronal
        model_eqn = [] #Lista donde se almacenarán las ecuaciones de la física que conocemos
        #Éste ciclo for solamente almacenará los elementos por encima de la diagonal (y la diagonal) de la matriz densidad   
        for i in range(self.dimension): #recorrer filas
            for j in range(i, self.dimension): #recorrer columnas
                if i == j:
                    #Si es un elemento de la diagonal, es real
                    model_eqn.append(f'{sympy.simplify(self.dot_rho[i,j])} - df({self.rho_sy[i,j]},t)')
                else: 
                    #Si es un elemento por encima de la diagonal
                    #Tomamos la parte real y la parte imaginaria de la componente de la matriz
                    model_eqn.append(f'{sympy.re(sympy.simplify(self.dot_rho[i,j]))} - df({sympy.re(self.rho_sy[i,j])},t)')            
                    model_eqn.append(f'{sympy.im(sympy.simplify(self.dot_rho[i,j]))} - df({sympy.im(self.rho_sy[i,j])},t) ')
        
        return model_eqn
    
    #Genera la ecuación simbólica Tr(rho) - 1 = 0, de la matriz densidad
    def density_matrix_trace(self):
        return f'{sympy.trace(self.rho_sy)-1}'    

class Solver(PhysicsEquations):
    def __init__(self, hamiltonian, control, init_control, init_condition, rho_target, neural_network_prop, L_operators=None, gammas=None):

        #Mediante la herencia de la clase PhysicsEquations, obtengo las ecuaciones físicas del sistema 
        super().__init__(hamiltonian, L_operators, gammas)

        #Configuración de la Red Neuronal
        self.pinn_prop = neural_network_prop #almacena configuración de la red neuronal
        self.train_states = {
            "init_condition": init_condition.astype(complex).tolist(),
            "rho_target": rho_target.astype(complex).tolist()
        } 
        self.system_props = {
            "hamiltonian": hamiltonian.tolist(),
            "control": list(control.items()),
            "L_operators": L_operators,
            "gammas": gammas
        }
        self.learning_rate = neural_network_prop["learning_rate"]
        self.epochs = int(neural_network_prop["epochs"])
        self.batch_size = int(neural_network_prop["batch_size"]) #Cantidad de elementos de muestra por lote
        self.neurons = int(neural_network_prop["neurons"])
        self.hid_layers = int(neural_network_prop["hidden_layers"])
        self.eta = neural_network_prop["eta"]
        self.eta_sc = neural_network_prop["eta_sc"]
        self.chi = neural_network_prop["chi"]

        #Se establece el hardware para usar en cálculos de la red neuronal 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando {self.device} como plataforma.")

        #Condición para realizar seguimiento del modelo en tiempo real
        self.debug_model = neural_network_prop["debug_model"]
        self.num_workers = neural_network_prop["num_workers"]
        
        #Condicion para implementar la pérdida del objetivo de la compuerta
        self.loss_gate = neural_network_prop["loss_gate"]
        
        #Valor mínimo para que se rompa el entrenamiento
        self.minLoss = 1e-6

        #Configuración del control
        self.control = control
        self.init_control = init_control
        self.control_len = len(self.control)
        self.init_control_keys = list(self.init_control.keys()) #Claves del control
        
        #Configuración de las condiciones iniciales
        self.init_condition = {i: matrix_flatten(elem) for i, elem in enumerate(init_condition.astype(complex))} #Agrupo estados iniciales en formato de diccionario enumerados con índice i
        self.init_condition_len = len(self.init_condition.values()) #Cantidad de estados iniciales
        self.init_condition_keys = list(self.init_condition[0].keys()) #Claves de las componentes
        self.rho_components = (self.dimension**2) #Cantidad de componentes elementos tomados de la matriz densidad por encima de la diagonal y la diagonal (parte real e imaginaria). 
        self.out_components = self.rho_components*self.init_condition_len + self.control_len #Cantidad de soluciones de salida de la PINN
        self.target_eqn = target_eqn_transform(self,rho_target, self.rho_sy) #Convierto estados objetivo en un diccionario aplanado 

        #Se define la configuración del vector discreto de tiempo
        self.t_0 = neural_network_prop["time_config"][0] #Tiempo inicial
        self.t_max = neural_network_prop["time_config"][1] #Tiempo final
        self.t_size = neural_network_prop["time_config"][2] #Número de elementos del vector de tiempo

        #Se define un vector columna de tiempo entre t_0 y t_max de t_size elementos y se agrega el efecto de una perturbación estocástica:
        self.t_grid = self.perturbar_puntos_tiempo(self.t_0, self.t_max, self.t_size)

        #Se crean minilotes de tiempo en formato dataset (en PINN la optimización no es muy significativa)
        self.dataset = TensorDataset(self.t_grid) #Datos de entrenamiento
        #self.batch_size = int(self.t_size/self.batch_number) #Cantidad de elementos de muestra por lote
        self.train_loader = DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True, prefetch_factor=None if self.num_workers == 0 else 2, persistent_workers=False if self.num_workers == 0 else True) #Datos de entrenamiento en lotes y barajados
        #num_workers > 0 es conveniente para GPUs.

        #Atributos que almacenan el historial de la función de perdida: total, por modelo, por objetivo, por restricciones
        self.loss_history = []
        self.loss_model = []
        self.loss_control = []
        self.loss_const = []
        self.loss_reg = []

        #Se genera una instancia de nuestro modelo de red neuronal
        self.modelo = NeuralNetwork(self.out_components, self.hid_layers, self.neurons).to(self.device)
        
        #Se hace una copia inicial del modelo de red neuronal.
        #Luego será actualizado por el modelo que minimice mejor el error
        self.modelo_optimo = copy.deepcopy(self.modelo).to(self.device)
    
    def parametric_solutions(self, t, modelo):
        """ Este método calcula las predicciones de la dinámica y el control"""
        ### Calcula la parametrización x(t) y u(t) usando la clave del lote + clave de la componente 
        
        #Evaluamos nuestra red neuronal en el vector de tiempo t
        N = modelo(t, self.init_condition_keys, self.init_control_keys, self.init_condition_len)

        #Obtenemos el diferencial entre el vector t y el valor t_0 (ocurre broadcasting)
        dt = t-self.t_0
        
        #Se define función ansatz que satisface adecuadamente las condiciones iniciales para x(t) y u(t)
        f = (1-torch.exp(-dt))
        
        #Tipo de funcion que satisface que el control inicie y termine en cero
        T_final=self.t_max #tiempo final
        
        def F1(dt,T):
            return (1-torch.exp(-dt))*(1-torch.exp(-(T-dt)))
        
        def F2(dt,T):
            return (1-torch.exp(-dt))*torch.exp(-dt/(0.1*T))
        
        #Se construyen las soluciones paramétricas para todas las condiciones iniciales del SISTEMA x(t)
        rho = {}
        for id in range(self.init_condition_len):
            rho[id] = {key: self.init_condition[id][key]+f*N[id][key] for key in self.init_condition_keys}
        
        #Se construyen las soluciones paramétricas para todas las condiciones iniciales del CONTROL u(t)  
        controls = {key: self.init_control[key] + F1(dt,T_final)*N[self.init_condition_len][key] for key in self.init_control_keys}
        
        return rho, controls
        
    def model_loss(self, t, rho, controls):
        """ Método que calcula la pérdida física, o pérdida del modelo. """
        # Términos de pérdida del modelo
        Lmodel = torch.tensor(0.).to(self.device)
        Lcontrol = torch.tensor(0.).to(self.device)
        Lconst = torch.tensor(0.).to(self.device)

        for id in range(self.init_condition_len):
            #Ecuaciones de la física
            for eqn in self.model_eqn:
                # Elevar al cuadrado el residuo de las ecuaciones en cada instante de tiempo y luego sacar el promedio es una forma de calcular error cuadrático medio
                Lmodel += eval(eqn, {"rho":rho[id], "df":self.df, "t":t, "ctl": controls}).pow(2).mean() 
                
            #Formulación del término de pérdida de la condición del estado objetivo
            if not self.loss_gate: 
                for L_i in self.target_eqn[id]:
                    Lcontrol += (eval(L_i, {"rho":rho[id]}).pow(2).mean())
            
            #if self.loss_gate:
            #                    
            #    t_T = torch.tensor([[self.t_max]], dtype=torch.float32).to(self.device)
            #    t_T.requires_grad = True
            #    rho_T_batch, _ = self.parametric_solutions(t_T, self.modelo)
            #    rho_T = rho_T_batch[id]  # mismo formato de antes
            #    for L_i in self.target_eqn[id]:
            #        result = eval(L_i, {"rho": rho_T})
            #        if not torch.is_tensor(result):
            #            result = torch.tensor(result, dtype=torch.float32, device=self.device)  # convierte escalar a tensor
            #        Lcontrol += result.pow(2).mean()
            
            if self.loss_gate:
                # Calcular siempre en el tiempo final T de forma directa (no desde el batch)
                T_tensor = torch.tensor([[self.t_max]], dtype=torch.float32, device=self.device)
                rho_T, _ = self.parametric_solutions(T_tensor, self.modelo)

                # rho_T es una lista de diccionarios por cada id
                rho_T = rho_T[id]  # estado final para este id

                for L_i in self.target_eqn[id]:
                    result = eval(L_i, {"rho": rho_T})
                    if not torch.is_tensor(result):
                        result = torch.tensor(result, dtype=torch.float32, device=self.device)
                    Lcontrol += result.pow(2).mean()

            #Formulación del término de pérdida de restricción de normalización de la matriz densidad
            Lconst += eval(self.trace_rho, {"rho":rho[id]}).pow(2).mean()

        #Función de costo total
        L = Lmodel + self.eta*Lcontrol + self.eta_sc*Lconst 
        return (L, Lmodel, self.eta*Lcontrol, self.eta_sc*Lconst)

    def df(self, f, t):
        return grad([f], [t], grad_outputs=(torch.ones(t.shape).to(self.device)), create_graph=True)[0]
        
    def perturbar_puntos_tiempo(self, t_0, t_max, t_size=300, sig=0.5):
        grid = torch.linspace(t_0, t_max, t_size).reshape(-1, 1)
        delta_t = grid[1] - grid[0]  
        noise = delta_t * torch.randn_like(grid)*sig
        t = grid + noise

        #Agrego puntos basura al grid de tiempo
        t.data[2] = torch.ones(1,1)*(-1)
        t.data[t<t_0]=t_0 - t.data[t<t_0]
        t.data[t>t_max]=2*t_max - t.data[t>t_max]
        t.data[0] = torch.ones(1,1)*t_0
        t.requires_grad = False
        return t

    def train_neural_network(self):
        """ Esta es la función principal que arranca el entrenamiento de la red neuronal """
        if not self.debug_model: print("Entrenando Red Neuronal...")
        
        #Se define el optimizador que actualiza los parámetros de la red, ej: Adam.
        optimizador = torch.optim.Adam(self.modelo.parameters(), lr=self.learning_rate, betas=[0.999, 0.9999]) 
        Llim =  1  #Valor de referencia

        for epoch in range(self.epochs):
            
            #Se establece el modelo en modo entrenamiento, es decir: pytorch optimiza el proceso de entramiento internamente
            self.modelo.train()

            #Definimos un contador de pérdida
            loss = 0
            loss_model = 0
            loss_control = 0
            loss_const = 0
            loss_reg = 0 

            #Realizamos el entrenamiento por cada minibatch (mini lote) de valores de tiempo
            for t_batch in self.train_loader:
                t_batch = t_batch[0].to(self.device) #Movemos los datos al GPU/CPU
                t_batch.requires_grad = True

                #Se evalúa la red neuronal con el mini lote de valores de tiempo t_batch
                #y se calculan las soluciones paramétricas de la ecuación maestra y del control
                rho, controls = self.parametric_solutions(t_batch, self.modelo)
        
                # Función de pérdida definida por las Ecs. de Hamilton (simplécticas): Escribiendo explícitamente las Ecs (más rápido)
                Ltot, Lmodel, Lcontrol, Lconst = self.model_loss(t_batch, rho, controls)
                
                l2_reg = torch.tensor(0.).to(self.device)

                for param in self.modelo.parameters(): #Recorro por cada capa los parámetros del modelo
                    l2_reg += torch.norm(param) #Calcula la norma L2 (o norma euclídea) del tensor de parámetros
                Ltot += self.chi * l2_reg

                #Backward realiza la retropropagación de la pérdida a través del modelo
                #Es el cálculo de los gradientes de la pérdida respecto a los parámetros del modelo
                #Backward hace el papel de una derivación con regla de la cadena mediante grafos computacionales
                Ltot.backward()
                
                #Se actualizan los parámetros del modelo mediante el modelo Adam (Adaptive Moment Estimation)
                optimizador.step()

                #Se reinician todos los gradientes calculados para todos los parámetros de la red
                optimizador.zero_grad()

                #Se convierte tensor de pérdida a valores escalares
                loss += Ltot.item()
                loss_model += Lmodel.item()
                loss_control += Lcontrol.item()
                loss_const += Lconst.item()
                loss_reg += (self.chi*l2_reg).item()
                #self.Lcontrol.append(Lcontrol.item())

            #Se almacena la pérdida por época
            self.loss_history.append(loss)  
            self.loss_model.append(loss_model)
            self.loss_control.append(loss_control)
            self.loss_const.append(loss_const)
            self.loss_reg.append(loss_reg)

            #Mantiene el mejor modelo (el de menor pérdida) utilizando deepcopy
            if  epoch > 0.8*self.epochs and Ltot < Llim:
                #Se crea una copia independiente del modelo con la menor pérdida
                self.modelo_optimo = copy.deepcopy(self.modelo).to(self.device)
                Llim=Ltot 

            if Ltot < self.minLoss:
                self.modelo_optimo = copy.deepcopy(self.modelo).to(self.device)
                print('Alcanzó el valor mínimo de pérdida')
                break
            
            if self.debug_model:
                clear_output(wait=True)
                print("Entrenando red neuronal...")
                print(f"Epoca: {epoch}, \nPérdida: {self.loss_history[-1]}")
                print(f"Lmodel: {self.loss_model[-1]}")
                print(f"Lcontrol: {self.loss_control[-1]}")
                print(f"Lconst: {self.loss_const[-1]}")
                print(f"Lreg: {self.loss_reg[-1]}")
    
        print("¡Proceso Finalizado!")
        
    #def plot_loss_curves(self):
    #    #Solamente se ejecuta en un notebook jupyter. Imprime la pérdida en función de las épocas
    #    fig_loss = plt.figure(figsize=(600/80, 300/80))  # Dividido por 80 para convertir píxeles a pulgadas (80 píxeles por pulgada es común)
    #    ax_loss = fig_loss.add_subplot(111)
    #    plt.yscale("log")
    #    #ax_loss.set_title('Términos de Pérdida', fontsize=16)
    #    ax_loss.set_xlabel('Epochs (log scale)')
    #    ax_loss.set_ylabel('Loss (log scale)')
    #    ax_loss.loglog(self.loss_history, 'b-', label=f"Total Loss Function")
    #    ax_loss.loglog(self.loss_model, 'r', label=f"Loss Model")
    #    ax_loss.loglog(self.loss_control, 'lime', label=f"Loss Control")
    #    ax_loss.loglog(self.loss_const, 'cyan', label=f"Loss Const")
    #    ax_loss.loglog(self.loss_reg, 'purple', label=f"Loss Reg")
    #    ax_loss.grid(True)
    #    ax_loss.legend()
    
    def plot_loss_curves(self):
        #Solamente se ejecuta en un notebook jupyter. Imprime la pérdida en función de las épocas
        fig_loss = plt.figure(figsize=(600/80, 300/80))  # Dividido por 80 para convertir píxeles a pulgadas (80 píxeles por pulgada es común)
        ax_loss = fig_loss.add_subplot(111)
        plt.yscale("log")
        #ax_loss.set_title('Términos de Pérdida', fontsize=16)
        ax_loss.set_xlabel('Epochs (log scale)')
        ax_loss.set_ylabel('Loss (log scale)')
        ax_loss.loglog(self.loss_history, 'b-', label=f"Total Loss Function | Min Loss: {self.loss_history[-1]:.2e}")
        ax_loss.loglog(self.loss_model, 'r', label=f"Loss Model          | Min Loss: {self.loss_model[-1]:.2e}")
        ax_loss.loglog(self.loss_control, 'lime', label=f"Loss Control        | Min Loss: {self.loss_control[-1]:.2e}")
        ax_loss.loglog(self.loss_const, 'cyan', label=f"Loss Const          | Min Loss: {self.loss_const[-1]:.2e}")
        ax_loss.loglog(self.loss_reg, 'purple', label=f"Loss Reg            | Min Loss: {self.loss_reg[-1]:.2e}")
        ax_loss.grid(True)
        ax_loss.legend(prop={'family': 'monospace', 'size': 9})

    def plot_loss(self, loss):
        #Solamente se ejecuta en un notebook jupyter. Imprime la pérdida en función de las épocas
        fig_loss = plt.figure(figsize=(600/80, 300/80))  # Dividido por 80 para convertir píxeles a pulgadas (80 píxeles por pulgada es común)
        ax_loss = fig_loss.add_subplot(111)
        plt.yscale("log")
        #ax_loss.set_title('Pérdida', fontsize=16)
        ax_loss.set_xlabel('Epochs (log scale)')
        ax_loss.set_ylabel('Loss (log scale)')
        ax_loss.loglog(loss, 'b-', label=f"Min Loss: {loss[-1]:.2e}")
        ax_loss.grid(True)
        ax_loss.legend()

    def eval_component(self, t, key, id=0, plot=False):
        """
        Evalúa la red neuronal en el tiempo t y retorna una componente de la matriz densidad.
        También puede retornar la función de control.

        :param t: Vector de tiempo.
        :type t: Tensor torch con gradientes
        :param key: Clave que identifica la componente de la matriz densidad/el control.
        :type key: Cadena de texto.
        :param id: Índice que identifica al estado inicial correspondiente.
        :type id: Entero.
        :param plot: Imprime la gráfica de la componente/función de control.
        :type plot: Booleano.
        
        :return: Evloución de la componente de la matriz densidad/Función de control.
        :rtype: Arreglo Numpy
        """
        self.modelo_optimo.eval()
        t = t.to(self.device)
        t.requires_grad=True

        rho_sol, controls_sol = self.parametric_solutions(t, self.modelo_optimo)
        component = torch.tensor(0.).to(self.device)

        try:
            component = rho_sol[id][key]
        except KeyError:
            component = controls_sol[key]     
        
        if plot:
            t_net = t.cpu().detach().numpy()
            # Plot vector components
            plt.figure(figsize=(5,4), tight_layout=True)
            plt.plot(t_net, component,'-b', linewidth=2, label=f'{key}'); 
            axes = plt.gca()
            plt.xlabel('t')
            axes.xaxis.label.set_size(14)
            axes.yaxis.label.set_size(14)
            plt.legend(prop={"size":10})
            plt.grid()

        return component.cpu().detach().numpy()
    
    def save_all(self, nombre_proyecto="Proyecto"):
        """
        Almacena toda la información de entrenamiento y el modelo de red neuronal entrenado.

        :param nombre_proyecto: Nombre del proyecto. Es un directorio en donde se guardarán todos los archivos.
        :type nombre_proyecto: Cadena de texto.
        """
        directorio = Path(f"Modelos/{nombre_proyecto}_{date.today()}")
        directorio.mkdir(parents=True, exist_ok=True)
        loss_history = {
            "loss_total": self.loss_history,
            "loss_model": self.loss_model,
            "loss_control": self.loss_control,
            "loss_const": self.loss_const,
            "loss_reg": self.loss_reg
        }

        torch.save(self.modelo_optimo.state_dict(), f"{directorio}/{nombre_proyecto}.pth")

        with open(f"{directorio}/loss_history.json", "w") as fichero:
            json.dump(loss_history, fichero)
        
        with open(f"{directorio}/pinn_prop.json", "w") as fichero:
            json.dump(self.pinn_prop, fichero)

        if isinstance(self.train_states, torch.Tensor):
            torch.save(self.train_states, f"{directorio}/train_states.pt")
        else:
            np.save(f"{directorio}/train_states.npy", self.train_states)
        
        with open(f"{directorio}/system_props.json", "w") as fichero:
            fichero.write(str(self.system_props))
        
def basis(dimension):    
    """
    Genero la base computacional kets de sistema de dimensón arbitraria 
    """
    base_kets = []
    for i in range(dimension):
        base_kets.append(sympy.Matrix(qutip.basis(dimension, i).full()))
    return base_kets

def lindbladian(a,b):
        r"""
        Generaliza el operador de lindblad para un par de operadores de disipación a y b:
        .. math::
            D(a,b)\rho = a \rho b^\dagger-\frac{1}{2}a^\dagger b\rho - \frac{1}{2}\rho a^\dagger b

        :param a: Operador de disipación "a"
        :type a: Sympy/Numpy object
        :param b: Operador de disipación "b"
        :type b: Sympy/Numpy object
        
        :return: Operador de Lindblad  
        :rtype: sympy objct
        """
        dimension = a.shape[0]
        rho = PhysicsEquations.density_matrix_create(dimension)
        return a*rho*sympy_dag(b) - 0.5*sympy_dag(a)*b*rho - 0.5*rho*sympy_dag(a)*b

def control_create(symbols, init_control):
    """ 
    Crea y retorna dos diccionarios: 
    1. El diccionario que contiene las funciones simbólicas de control
    2. El diccionario que contiene los valores iniciales de control
    """
    return [{key: sympy.symbols(f"ctl['{key}']", real=True) for key in symbols}, {key: val for key, val in zip(symbols, init_control)}]


def matrix_flatten(matrix):
    """ 
    Función que toma una matriz y arroja un diccionario o un arreglo con las componentes por encima de la diagonal 
    (y su diagonal). Si es un diccionario:
    - Las claves corresponden a las etiquetas de cada componente, e.g: rho_00 
    - Los valores corresponden a las componentes de la matriz (parte real e imaginaria)
    """
    def real(val):
        return sympy.re(val) if isinstance(val, sympy.Expr) else np.real(val) 
            
    def imag(val):
        return sympy.im(val) if isinstance(val, sympy.Expr) else np.imag(val) 
            

    m_flatten = {}
    for i in range(matrix.shape[0]): #Recorre filas
        for j in range(i, matrix.shape[1]): #Recorre columnas
            if i == j:
                #Si es un elemento dela diagonal, es real
                m_flatten[f"rho_{i}{j}"] = real(matrix[i,j])
            else: 
                #Si es un elemento por encima de la diagonal
                #Tomamos la parte real y la parte imaginaria de la componente de la matriz
                m_flatten[f"rho_{i}{j}"] = real(matrix[i,j])
                m_flatten[f"rho_{i}{j}i"] = imag(matrix[i,j])
    
    return m_flatten
    
def target_eqn_transform(self,rho_target, rho_sy):
    """, rho_sy, rho_target 
    Toma una matriz que contiene los estados objetivos impuestos al sistema físico. Para cada estado objetivo denotado por el *id*, se calcula la diferencia o residuo con respecto a la matriz densidad. La matriz resultante se aplana en formato diccionario, donde cada clave denota la posición del elemento en su matriz de origen. Como cada elemento de la nueva matriz es una ecuación igualada a 0, se procede a convertirla en formato de texto. 
    """
    res = {}
    for id, matrix in enumerate(rho_target):
        m_flat = matrix_flatten(rho_sy-matrix)
        res[id] = [str(values) for values in m_flat.values()]
    return res