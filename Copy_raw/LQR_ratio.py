"""
LQR控制器
"""
import numpy as np
from data_structures import Car_data, Car_State

class LQR_car:
    def __init__(self, max_num_iteration=700, tolerance=1e-6, dt=0.1):
        self.max_num_iteration = max_num_iteration
        self.tolerance = tolerance
        self.dt = dt
        self.A = np.zeros((2, 2))
        self.B = np.zeros((2, 2))
        self.Q = np.zeros((2, 2))
        self.R = np.zeros((2, 2))
        self.M = np.zeros((2, 2))
        self.P = np.zeros((2, 2))
        self.cur_state = Car_State()
        self.CarData = Car_data()

    def update_car_state(self, x, y, psi, v):
        self.cur_state.x = x
        self.cur_state.y = y
        self.cur_state.psi = psi
        self.cur_state.v = v

    def Update_A_B_matrix(self, L11):
        v = self.cur_state.v
        dt = self.dt
        self.A[0, 0] = 1.0
        self.A[0, 1] = 0.0
        self.A[1, 0] = v * dt
        self.A[1, 1] = 1.0
        
        # B matrix: [v*dt/L, 0]^T
        self.B[0, 0] = v * dt / L11  # affects e_psi
        self.B[1, 0] = 0.0           # affects e_y (directly 0)


    def Update_Q_R_matrix(self, q11, q22, r00, r11, heading):
        q1 = q11
        q2 = q22
        r = r00
        if heading == 1:
            v = abs(self.cur_state.v)
            if 0.2 < v < 0.75:
                q2 = 135; r = 70
            elif 0.75 <= v < 1.2:
                q2 = 115; r = 80
            elif 1.2 <= v < 1.75:
                q2 = 125; r = 80
            elif 1.75 <= v < 2.0:
                q2 = 120; r = 95
            elif 2.0 <= v < 2.5:
                q2 = 110; r = 120
            elif 2.5 <= v < 3.0:
                q2 = 95; r = 145
            elif 3.0 <= v < 3.5:
                q2 = 75; r = 175
            elif v >= 3.5:
                q2 = 60; r = 200
        self.Q[0, 0] = q1
        self.Q[1, 1] = q2
        self.R[0, 0] = r
        self.R[1, 1] = r11

    def _solve_riccati(self, A, B, Q, R, M=None):
        if M is None:
            M = np.zeros_like(Q)
        P = Q.copy()
        AT = A.T
        BT = B.T
        MT = M.T
        for _ in range(self.max_num_iteration):
            temp1 = AT @ P @ B + M
            temp2 = R + BT @ P @ B
            temp2_inv = np.linalg.inv(temp2)
            temp3 = BT @ P @ A + MT
            P_next = AT @ P @ A - temp1 @ temp2_inv @ temp3 + Q
            diff = np.max(np.abs(P_next - P))
            P = P_next
            if diff < self.tolerance:
                break
        K = np.linalg.inv(R + BT @ P @ B) @ (BT @ P @ A + MT)
        return K, P

    def Solve(self, A=None, B=None, Q=None, R=None, M=None, ptr_K=None):
        if A is None:
            K, P = self._solve_riccati(self.A, self.B, self.Q, self.R, self.M)
        else:
            if M is None:
                M = np.zeros_like(Q)
            K, P = self._solve_riccati(A, B, Q, R, M)
        if ptr_K is not None:
            if isinstance(ptr_K, list) and len(ptr_K) > 0:
                ptr_K[0] = K
            else:
                ptr_K[:] = K
        return K, P

    def CALC(self, dif_state, ptr_K=None):
        if isinstance(dif_state, list):
            dif_state = np.array(dif_state).reshape(2, 1)
        elif dif_state.shape == (2,):
            dif_state = dif_state.reshape(2, 1)
        K, _ = self.Solve(ptr_K=ptr_K)
        U = -K @ dif_state
        self.CarData.car_K0 = K[0, 0]
        self.CarData.car_K1 = K[0, 1] if K.shape[1] > 1 else 0.0
        self.CarData.car_U0 = U[0, 0] if U.size > 0 else 0.0
        return self.CarData.car_U0