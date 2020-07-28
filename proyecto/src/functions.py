import numpy as np
from copy import copy
import rbdl

pi = np.pi

class Robot(object):
    def __init__(self, q0, dq0, ndof, dt):
        self.q = q0    # numpy array (ndof x 1)
        self.dq = dq0  # numpy array (ndof x 1)
        self.M = np.zeros([ndof, ndof])
        self.b = np.zeros(ndof)
        self.dt = dt
        self.robot = rbdl.loadModel('../urdf/robot.urdf')

    def send_command(self, tau):
        rbdl.CompositeRigidBodyAlgorithm(self.robot, self.q, self.M)
        rbdl.NonlinearEffects(self.robot, self.q, self.dq, self.b)
        ddq = np.linalg.inv(self.M).dot(tau-self.b)
        self.q = self.q + self.dt*self.dq
        self.dq = self.dq + self.dt*ddq

    def read_joint_positions(self):
        return self.q

    def read_joint_velocities(self):
        return self.dq


def dh(d, theta, a, alpha):
    """
    Matriz de transformacion homogenea asociada a los parametros DH.
    Retorna una matriz 4x4
    """
    sth = np.sin(theta)
    cth = np.cos(theta)
    sa  = np.sin(alpha)
    ca  = np.cos(alpha)
    T = np.array([[cth, -ca*sth,  sa*sth, a*cth],
                   [sth,  ca*cth, -sa*cth, a*sth],
                   [0.0,      sa,      ca,     d],
                   [0.0,     0.0,     0.0,   1.0]])
    return T

def dh_2(d, theta, a, alpha):
    """
    Matriz de transformacion homogenea asociada a los parametros DH.
    Retorna una matriz 4x4
    """
    sth = np.sin(theta)
    cth = np.cos(theta)
    sa  = np.sin(alpha)
    ca  = np.cos(alpha)
    T = np.matrix([[cth, -ca*sth,  sa*sth, a*cth],
                   [sth,  ca*cth, -sa*cth, a*sth],
                   [0.0,      sa,      ca,     d],
                   [0.0,     0.0,     0.0,   1.0]])
    return T

def robot_fkine(q):
    """
    Calcular la cinematica directa del robot UR5 dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5]
    """
    # Matrices DH
    T1 = dh( 0.07,        q[0],     0.01, pi/2)
    T2 = dh(      0, q[1]+pi/2, 0.105,    0)
    T3 = dh(      0,        q[2], 0.097,    0)
    T4 = dh(      0, q[3],     0.03, pi/2)
    T5 = dh(0.16,     q[4],     0, 0)
    # Efector final con respecto a la base
    T = T1.dot(T2).dot(T3).dot(T4).dot(T5)
    return T

def robot_fkine_m(q):
    """
    Calcular la cinematica directa del robot UR5 dados sus valores articulares. 
    q es un vector numpy de la forma [q1, q2, q3, q4, q5]
    """
    # Matrices DH
    T11 = dh_2( 0.07,        q[0],     0.01, pi/2)
    T22 = dh_2(      0, q[1]+pi/2, 0.105,    0)
    T33 = dh_2(      0,        q[2], 0.097,    0)
    T44 = dh_2(      0, q[3],     0.03, pi/2)
    T55 = dh_2(0.16,     q[4],     0, 0)
    # Efector final con respecto a la base
    TT = T11.dot(T22).dot(T33).dot(T44).dot(T55)
    return TT

def ikine_robot(xdes, q0):
    """
    Calcular la cinematica inversa de UR5 numericamente a partir de la configuracion articular inicial de q0. 
    """
    epsilon  = 0.001
    max_iter = 1000
    delta    = 0.00001

    q  = copy(q0)
    for i in range(max_iter):
        # Main loop
        J = jacobian(q,delta) # Jacobiano
        T = robot_fkine(q) # Cinematica directa
        F = T[np.arange(3),3] # T de Posicion
        e = xdes - F # Obtencion de error 
        q = q + np.dot(np.linalg.pinv(J),e)  # Feedback
        if (np.linalg.norm(e) < epsilon):
            break # Cierre por condicion de error minimo
    return q

def jacobian(q, delta=0.0001):
    """
    Jacobiano analitico para la posicion. Retorna una matriz de 3x6 y toma como
    entrada el vector de configuracion articular q=[q1, q2, q3, q4, q5]
    """
    # Crear una matriz 3x5
    J = np.zeros((3,5))
    # Transformacion homogenea inicial (usando q)
    T = robot_fkine(q)
    # Iteracion para la derivada de cada columna
    for i in xrange(5):
        # Copiar la configuracion articular inicial
        dq = copy(q)
        # Incrementar la articulacion i-esima usando un delta
        dq[i] = dq[i] + delta
        # Transformacion homogenea luego del incremento (q+delta)
        Td = robot_fkine(dq)
        # Aproximacion del Jacobiano de posicion usando diferencias finitas
        for e in xrange(3):
            J[e][i] = ((Td[e][3]-T[e][3])/delta) 
    return J