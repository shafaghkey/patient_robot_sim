"""humanROM_utils

Human range-of-motion (ROM) enforcement utilities using One-Class SVM models
trained on subject-specific motion-capture data.

Classes:
  human_ROM_2d    — 2-D ROM constraint (shoulder/elbow angles)
  human_ROM_4d    — 4-D ROM constraint; loads joint-angle data from
                    agents/patient_svm_model/subject<N>_<cond>_HUB_angles.npz
  human_ROM_SVM_npz — loads pre-trained SVM model directly from an .npz file
                      (agents/patient_svm_model/subject<N>_<arm>_svm_model.npz)

Used by robot_patient_pybullet simulation modules and hri_sim_ros_patient_impaired.
"""
import pybullet as p
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import os
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler


# find file path from another package
def find_file_path(package_name, relative_path):
    """Find the absolute file path given a package name and a relative path within that package."""
    try:
        import rospkg
        rospack = rospkg.RosPack()
        package_path = rospack.get_path(package_name)
        full_path = os.path.join(package_path, relative_path)
        if os.path.exists(full_path):
            return full_path
        else:
            raise FileNotFoundError(f"File not found: {full_path}")
    except Exception as e:
        print(f"Error finding file path: {e}")
        return None

class human_ROM_2d:
    def __init__(self, subject=3, impaired_arm=None, idx_to_use=[1,3]):
        """Enforce ROM joint limits for human"""
        self.svm_model_impaired = None
        self.svm_model_healthy = None
        self.scaler_impaired = None
        self.scaler_healthy = None
        self.idx_to_use = idx_to_use  # which indices of the 4D angles to use for 2D SVM
        # self.support_vectors = None
        # self.alphas = None
        # self.gamma = None
        # self.bias = None
        
        if impaired_arm == "R": 
            healthy_arm = "L"
        elif impaired_arm == "L":
            healthy_arm = "R"
        else:
            raise ValueError("Invalid impaired arm. Choose 'R' or 'L'.")
        
        self.subject = subject
        self.impaired_arm = impaired_arm
        self.healthy_arm = healthy_arm

    def train_svm_model(self, kernel='rbf', sigma=20.0, nu=0.001, scaled=True):
        """Train SVM model on angles for the specified arm.
        Args:
            arm (str): 'R' for right arm, 'L' for left arm.
            kernel (str): SVM kernel type.
            sigma (float): Sigma for RBF kernel.
            nu (float): SVM parameter nu.
            scaled (bool): Whether to scale the angles.
        Returns:
            svm_model (OneClassSVM): Trained SVM model.
            scaler (StandardScaler): Scaler used for scaling angles, if scaled=True.
        """
        for condition in ['impaired', 'healthy']:
            arm = self.impaired_arm if condition == 'impaired' else self.healthy_arm
            print(f"Training SVM model for {arm}_{condition} arm angles...")
            time_start = time.time()
            rom_data = np.load(f"agents/patient_svm_model/subject{self.subject}_{condition}_HUB_angles.npz", allow_pickle=True)
            rom_data = {key: rom_data[key] for key in rom_data.files}
            eul_right, eul_left = rom_data['eul_right'], rom_data['eul_left']
            eul = eul_right if arm == "R" else eul_left

            chain_R = ['Skeleton', 'Ab', 'Chest','RShoulder','RUArm','RFArm','RHand']
            HUB_dof_R = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',   
                        'Ab_bending', 'Ab_rotation', 'Ab_twist',    
                        'Chest_bending', 'Chest_rotation', 'Chest_twist', 
                        'RClavicle Protraction', 'RClavicle Elevation','RClavicle Axial Rotation',     
                        'Shoulder Abduction', 'Shoulder Extension', 'Shoulder Rotation',
                        'Elbow Flexion', 'Elbow_0', 'Elbow Supination',    
                        'Wrist Deviation', 'Wrist Supination', 'Wrist Flexion']

            sh_abd = (eul[:,4,0])
            sh_ext = (eul[:,4,1])
            sh_rot = (eul[:,4,2])
            el_fle = (eul[:,5,0])

            angle_labels = ['Shoulder Abduction (deg)', 'Shoulder Extension (deg)', 'Shoulder Rotation (deg)', 'Elbow Flexion (deg)']
            angles_svm = np.column_stack((sh_abd, sh_ext, sh_rot, el_fle))
            angles_svm = angles_svm[~np.isnan(angles_svm).any(axis=1)]  # Remove rows with NaN values

            angle_labels = [angle_labels[i] for i in self.idx_to_use]
            angles_svm = angles_svm[:, self.idx_to_use]
            print(f"Angles for 2D version: {angle_labels}")
            print(f"Angles SVM shape after NaN removal: {angles_svm.shape}")

            if scaled:
                scaler = StandardScaler()
                angles_svm = scaler.fit_transform(angles_svm)
            else:
                scaler = None

            # model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.001, verbose=True) #gamma = 1 / (n_features * X.var())
            sigma = 2.0
            gamma = 1 / (2 * sigma ** 2)      # Calculate gamma from sigma
            svm_model = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu, verbose=True)

            svm_model.fit(angles_svm)
            # self.support_vectors = self.svm_model.support_vectors_
            # self.alphas = self.svm_model.dual_coef_[0]
            # self.bias = self.svm_model.intercept_[0] #if hasattr(model, 'intercept_') else np.mean(np.abs(model.decision_function(angles_scaled) - 1))

            # gamma_scale = 1 / (angles_svm.shape[1] * angles_svm.var())  # gamma = 1 / (n_features * X.var())
            # sigma_scale = 1 / np.sqrt(2 * gamma_scale)  # Calculate sigma from gamma
            # print(f"Gamma (scale): {gamma_scale:.4f}, Sigma (scale): {sigma_scale:.4f}, nu: 0.001")
            svm_train_time = time.time() - time_start
            print(f"SVM model for {arm}_{condition} arm trained in {svm_train_time:.3f} seconds")
            print(f"ROM svm model: gamma: {gamma:.4f}, sigma: {sigma:.2f}, nu: {nu:.4f}, bias: {svm_model.intercept_[0]:.2f}")
            
            # predictions and max decision function value
            decision_function = svm_model.decision_function(angles_svm)
            max_decision_value = np.max(decision_function)
            print(f"Max decision function value of {arm}_{condition} arm: {max_decision_value:.4f}")

            if condition == "impaired":
                self.svm_model_impaired, self.scaler_impaired = svm_model, scaler
            else:
                self.svm_model_healthy, self.scaler_healthy = svm_model, scaler
        # return svm_model, scaler

    def calc_Gamma_and_derivative (self, x, condition='impaired'):  # = model.decision_function(x)[0]
        """Calculate the decision function for the SVM model."""
        #transform using the scaler_scale&mean
        # x = (x - self.scaler_mean) / self.scaler_scale
        # x = x / self.scaler_scale
        if condition == 'impaired':
            svm_model = self.svm_model_impaired
            scaler = self.scaler_impaired
        else:
            svm_model = self.svm_model_healthy
            scaler = self.scaler_healthy

        support_vectors = svm_model.support_vectors_
        alphas = svm_model.dual_coef_[0]
        gamma = svm_model.gamma
        bias = svm_model.intercept_[0]

        if scaler is not None:
            x = (x - scaler.mean_) / scaler.scale_
            # x = scaler.fit_transform(x.reshape(1, -1))[0]  # Scale the input x
            
        Gamma_val = 0
        Gamma_grad = np.zeros_like(x)
        for i, sv in enumerate(support_vectors):
            Gamma_val += np.exp(-gamma * np.linalg.norm(x - sv) ** 2) * alphas[i]
            Gamma_grad += -2 * gamma * np.exp(-gamma * np.linalg.norm(x - sv) ** 2) * alphas[i] * (x - sv)
        Gamma_val += bias
        # decision_value = svm_model.decision_function(x.reshape(1, -1))[0]
        # print(f"Decision value: {decision_value:.4f}, Gamma: {Gamma_val:.4f}, Gamma_grad: {Gamma_grad}")
        return Gamma_val, Gamma_grad
    
    def enforce_ROM_tau(self, q):
        """Enforce ROM joint limits for human"""
        tau = np.zeros_like(q)
        # q = q[:4]  # Use only the first 4 joints (shoulder, elbow_flexion)
        Gamma, Gamma_grad = self.calc_Gamma_and_derivative(q)
        if Gamma < -0.05:  # if the point is inside the ROM
            # If the point is outside the ROM, apply a torque to bring it back inside
            tau = - Gamma * Gamma_grad
        elif Gamma > -0.05 and Gamma < 0.05:
            tau = - Gamma * Gamma_grad
        return tau
    
class human_ROM_4d:
    def __init__(self, subject=3, impaired_arm=None):
        """Enforce ROM joint limits for human"""
        self.svm_model_impaired = None
        self.svm_model_healthy = None
        self.scaler_impaired = None
        self.scaler_healthy = None
        # self.support_vectors = None
        # self.alphas = None
        # self.gamma = None
        # self.bias = None
        
        if impaired_arm == "R": 
            healthy_arm = "L"
        elif impaired_arm == "L":
            healthy_arm = "R"
        else:
            raise ValueError("Invalid impaired arm. Choose 'R' or 'L'.")
        
        self.subject = subject
        self.impaired_arm = impaired_arm
        self.healthy_arm = healthy_arm

    def train_svm_model(self, kernel='rbf', sigma=2.0, nu=0.001, scaled=True):
        """Train SVM model on angles for the specified arm.
        Args:
            arm (str): 'R' for right arm, 'L' for left arm.
            kernel (str): SVM kernel type.
            sigma (float): Sigma for RBF kernel.
            nu (float): SVM parameter nu.
            scaled (bool): Whether to scale the angles.
        Returns:
            svm_model (OneClassSVM): Trained SVM model.
            scaler (StandardScaler): Scaler used for scaling angles, if scaled=True.
        """
        for condition in ['impaired', 'healthy']:
            arm = self.impaired_arm if condition == 'impaired' else self.healthy_arm
            print(f"Training SVM model for {arm}_{condition} arm angles...")
            time_start = time.time()
            rom_data = np.load(f"agents/patient_svm_model/subject{self.subject}_{condition}_HUB_angles.npz", allow_pickle=True)
            rom_data = {key: rom_data[key] for key in rom_data.files}
            eul_right, eul_left = rom_data['eul_right'], rom_data['eul_left']
            eul = eul_right if arm == "R" else eul_left

            chain_R = ['Skeleton', 'Ab', 'Chest','RShoulder','RUArm','RFArm','RHand']
            HUB_dof_R = ['pelvis_tilt', 'pelvis_list', 'pelvis_rotation',   
                        'Ab_bending', 'Ab_rotation', 'Ab_twist',    
                        'Chest_bending', 'Chest_rotation', 'Chest_twist', 
                        'RClavicle Protraction', 'RClavicle Elevation','RClavicle Axial Rotation',     
                        'Shoulder Abduction', 'Shoulder Extension', 'Shoulder Rotation',
                        'Elbow Flexion', 'Elbow_0', 'Elbow Supination',    
                        'Wrist Deviation', 'Wrist Supination', 'Wrist Flexion']

            sh_abd = (eul[:,4,0])
            sh_ext = (eul[:,4,1])
            sh_rot = (eul[:,4,2])
            el_fle = (eul[:,5,0])

            angle_labels = ['Shoulder Abduction (deg)', 'Shoulder Extension (deg)', 'Shoulder Rotation (deg)', 'Elbow Flexion (deg)']
            angles_svm = np.column_stack((sh_abd, sh_ext, sh_rot, el_fle))
            angles_svm = angles_svm[~np.isnan(angles_svm).any(axis=1)]  # Remove rows with NaN values

            if scaled:
                scaler = StandardScaler()
                angles_svm = scaler.fit_transform(angles_svm)
            else:
                scaler = None

            # model = OneClassSVM(kernel='rbf', gamma='scale', nu=0.001, verbose=True) #gamma = 1 / (n_features * X.var())
            sigma = 2.0
            gamma = 1 / (2 * sigma ** 2)      # Calculate gamma from sigma
            svm_model = OneClassSVM(kernel='rbf', gamma=gamma, nu=nu, verbose=True)

            svm_model.fit(angles_svm)
            # self.support_vectors = self.svm_model.support_vectors_
            # self.alphas = self.svm_model.dual_coef_[0]
            # self.bias = self.svm_model.intercept_[0] #if hasattr(model, 'intercept_') else np.mean(np.abs(model.decision_function(angles_scaled) - 1))

            # gamma_scale = 1 / (angles_svm.shape[1] * angles_svm.var())  # gamma = 1 / (n_features * X.var())
            # sigma_scale = 1 / np.sqrt(2 * gamma_scale)  # Calculate sigma from gamma
            # print(f"Gamma (scale): {gamma_scale:.4f}, Sigma (scale): {sigma_scale:.4f}, nu: 0.001")
            svm_train_time = time.time() - time_start
            print(f"SVM model for {arm}_{condition} arm trained in {svm_train_time:.3f} seconds")
            print(f"ROM svm model: gamma: {gamma:.4f}, sigma: {sigma:.2f}, nu: {nu:.4f}, bias: {svm_model.intercept_[0]:.2f}")
            
            # predictions and max decision function value
            decision_function = svm_model.decision_function(angles_svm)
            max_decision_value = np.max(decision_function)
            print(f"Max decision function value of {arm}_{condition} arm: {max_decision_value:.4f}")

            if condition == "impaired":
                self.svm_model_impaired, self.scaler_impaired = svm_model, scaler
            else:
                self.svm_model_healthy, self.scaler_healthy = svm_model, scaler
        # return svm_model, scaler

    def calc_Gamma_and_derivative (self, x, condition='impaired'):  # = model.decision_function(x)[0]
        """Calculate the decision function for the SVM model."""
        #transform using the scaler_scale&mean
        # x = (x - self.scaler_mean) / self.scaler_scale
        # x = x / self.scaler_scale
        if condition == 'impaired':
            svm_model = self.svm_model_impaired
            scaler = self.scaler_impaired
        else:
            svm_model = self.svm_model_healthy
            scaler = self.scaler_healthy

        support_vectors = svm_model.support_vectors_
        alphas = svm_model.dual_coef_[0]
        gamma = svm_model.gamma
        bias = svm_model.intercept_[0]

        if scaler is not None:
            x = (x - scaler.mean_) / scaler.scale_
            # x = scaler.fit_transform(x.reshape(1, -1))[0]  # Scale the input x
            
        Gamma_val = 0
        Gamma_grad = np.zeros_like(x)
        for i, sv in enumerate(support_vectors):
            Gamma_val += np.exp(-gamma * np.linalg.norm(x - sv) ** 2) * alphas[i]
            Gamma_grad += -2 * gamma * np.exp(-gamma * np.linalg.norm(x - sv) ** 2) * alphas[i] * (x - sv)
        Gamma_val += bias
        # decision_value = svm_model.decision_function(x.reshape(1, -1))[0]
        # print(f"Decision value: {decision_value:.4f}, Gamma: {Gamma_val:.4f}, Gamma_grad: {Gamma_grad}")
        return Gamma_val, Gamma_grad
    
    def enforce_ROM_tau(self, q):
        """Enforce ROM joint limits for human"""
        tau = np.zeros_like(q)
        # q = q[:4]  # Use only the first 4 joints (shoulder, elbow_flexion)
        Gamma, Gamma_grad = self.calc_Gamma_and_derivative(q)
        if Gamma < -0.05:  # if the point is inside the ROM
            # If the point is outside the ROM, apply a torque to bring it back inside
            tau = - Gamma * Gamma_grad
        elif Gamma > -0.05 and Gamma < 0.05:
            tau = - Gamma * Gamma_grad
        return tau
    

class human_ROM_SVM_npz:
    def __init__(self, svm=None):
        """Enforce ROM joint limits for human"""
        if svm is None:
            # Load the SVM model for a specific subject and arm
            subject, arm = 3, "impaired"  # Subject 3, impaired arm= right
            svm = np.load(f"agents/patient_svm_model/subject{subject}_{arm}_svm_model.npz", allow_pickle=True)
        # for key in svm.keys():
        #     print(f"svm[{key}]: {svm[key].shape if isinstance(svm[key], np.ndarray) else svm[key]}")
        self.scaler = svm['scaler']
        self.scaler_scale = svm['scaler_scale']
        self.scaler_mean = svm['scaler_mean']
        self.bias = svm['bias']
        self.support_vectors = svm['support_vectors']
        self.alphas = svm['alphas']
        self.gamma = svm['gamma']

        print(f"ROM svm model: gamma={self.gamma}, bias={self.bias}")

    def calc_Gamma (self, x):  # = model.decision_function(x)[0]
        """Calculate the decision function for the SVM model."""
        #transform using the scaler_scale&mean
        # x = (x - self.scaler_mean) / self.scaler_scale
        # x = x / self.scaler_scale
        Gamma_val = 0
        for i, sv in enumerate(self.support_vectors):
            dist = np.linalg.norm(sv - x)
            Gamma_val += np.exp(-self.gamma * dist ** 2) * self.alphas[i]
        Gamma_val += self.bias
        return Gamma_val

    def calc_Gamma_derivative (self, x):
        """Calculate the decision function for the SVM model."""
        # x = (x - self.scaler_mean) / self.scaler_scale
        Gamma_grad = np.zeros_like(x)
        for i, sv in enumerate(self.support_vectors):
            dist = np.linalg.norm(sv - x)
            Gamma_grad += -2 * self.gamma * np.exp(-self.gamma * dist ** 2) * self.alphas[i] * (x - sv)
        return Gamma_grad
    
    def enforce_ROM_tau(self, q):
        """Enforce ROM joint limits for human"""
        tau = np.zeros_like(q)
        # q = q[:4]  # Use only the first 4 joints (shoulder, elbow_flexion)
        Gamma = self.calc_Gamma(q)
        if Gamma < 0:  # if the point is inside the ROM
            # If the point is outside the ROM, apply a torque to bring it back inside
            tau = - Gamma * self.calc_Gamma_derivative(q)
        return tau