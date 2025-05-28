import torch
import torch.nn as nn
import numpy as np
from smplx import SMPLX
from lib.utils.geometry import rotation_matrix_to_angle_axis, rot6d_to_rotmat
from lib.models.smpl import H36M_TO_J14
from lib.models.projector import projection
from lib.core.config import VIBE_DATA_DIR

class SMPLXRegressor(nn.Module):
    def __init__(self, smplx_model_dir=VIBE_DATA_DIR, smplx_mean_params='data/vibe_data/smplx_mean_params.npz', num_betas=10):
        super(SMPLXRegressor, self).__init__()
        
        self.num_betas = num_betas
        self.npose = 24 * 6  

        # 网络结构
        self.fc1 = nn.Linear(512 * 4 + self.npose + 13, 1024)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(1024, 1024)
        self.drop2 = nn.Dropout()

        # 输出头 (确保输出维度正确)
        self.decpose = nn.Linear(1024, self.npose)  # 21×6=126
        self.decshape = nn.Linear(1024, num_betas)
        self.deccam = nn.Linear(1024, 3)
        self.decexp = nn.Linear(1024, 10)              # 新增：表情参数（10维）
        self.decjaw = nn.Linear(1024, 3)               # 新增：下巴姿态
        self.decleye = nn.Linear(1024, 3)              # 新增：左眼姿态
        self.decreye = nn.Linear(1024, 3)              # 新增：右眼姿态
        # 初始化SMPLX模型 (关键修改)
        self.smplx = SMPLX(
            model_path=smplx_model_dir,
            create_global_orient=True,
            create_body_pose=True,
            create_betas=True,
            create_left_hand_pose=True,
            create_right_hand_pose=True,
            create_expression=True,  # 根据需求调整
            create_jaw_pose=True,   
            create_leye_pose=True,
            create_reye_pose=True,
            use_pca=False,
            flat_hand_mean=True,      # 重要参数
            #batch_size=64
        )

        # 初始化参数
        mean_params = np.load(smplx_mean_params)
        init_pose = mean_params['pose'][:self.npose]  # [144]
        self.register_buffer('init_jaw', torch.zeros(1, 3))
        self.register_buffer('init_leye', torch.zeros(1, 3))  # 新增：左眼初始化
        self.register_buffer('init_reye', torch.zeros(1, 3))  # 新增：右眼初始化 
        self.register_buffer('init_exp', torch.zeros(1, 10)) 
        self.register_buffer('init_pose', torch.from_numpy(init_pose).float().unsqueeze(0))  # [1, 144]
        self.register_buffer('init_shape', torch.from_numpy(mean_params['shape'][:num_betas]).float().unsqueeze(0))  # [1, 10]
        self.register_buffer('init_cam', torch.from_numpy(mean_params['cam'][:3]).float().unsqueeze(0))  # [1, 3]
        self.register_buffer('init_lhand', torch.zeros(1, 45))
        self.register_buffer('init_rhand', torch.zeros(1, 45))

    def forward(self, x, init_pose=None, init_shape=None, init_cam=None, n_iter=3, J_regressor=None):
        batch_size = x.shape[0]

        # 初始化参数 (确保维度正确)
        pred_pose = self.init_pose.expand(batch_size, -1) if init_pose is None else init_pose
        pred_shape = self.init_shape.expand(batch_size, -1) if init_shape is None else init_shape
        pred_cam = self.init_cam.expand(batch_size, -1) if init_cam is None else init_cam
        pred_exp = self.init_exp.expand(batch_size, -1)
        pred_jaw = self.init_jaw.expand(batch_size, -1)
        pred_leye = self.init_leye.expand(batch_size, -1)  # 左眼姿态
        pred_reye = self.init_reye.expand(batch_size, -1)  # 右眼姿态
        pred_lhand = self.init_lhand.expand(batch_size, -1)
        pred_rhand = self.init_rhand.expand(batch_size, -1)

        # 迭代优化
        for _ in range(n_iter):
            xc = torch.cat([x, pred_pose, pred_shape, pred_cam], 1)
            xc = self.fc1(xc)
            xc = self.drop1(xc)
            xc = self.fc2(xc)
            xc = self.drop2(xc)
            
            pred_pose = self.decpose(xc) + pred_pose
            pred_shape = self.decshape(xc) + pred_shape
            pred_cam = self.deccam(xc) + pred_cam
            pred_exp = self.decexp(xc) + pred_exp
            pred_jaw = self.decjaw(xc) + pred_jaw
            pred_leye = self.decleye(xc) + pred_leye  # 更新左眼
            pred_reye = self.decreye(xc) + pred_reye  # 更新右眼

        # 关键修改：正确的姿态转换
        n_joints = self.npose // 6  # 24
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(batch_size, n_joints, 3, 3)
    
    # 分离全局旋转和身体姿态
        global_orient_mat = pred_rotmat[:, 0]  # [batch, 3, 3]
        global_orient_aa = rotation_matrix_to_angle_axis(global_orient_mat)  # [batch, 3]
        body_pose = pred_rotmat[:, 1:22]   # 只取1~21号关节 [batch, 21, 3, 3]
    
    # 关键修改：正确的轴角转换
        body_pose_flat = body_pose.reshape(-1, 3, 3)  # [batch*21, 3, 3]
        body_pose_aa = rotation_matrix_to_angle_axis(body_pose_flat)  # [batch*21, 3]
        body_pose_aa = body_pose_aa.reshape(batch_size, 21*3)  # [batch, 63]
    
    # 验证尺寸
        assert pred_shape.shape[0] == batch_size
        assert body_pose_aa.shape[0] == batch_size
        assert global_orient_aa.shape[0] == batch_size
        assert pred_exp.shape[0] == batch_size
        assert pred_jaw.shape[0] == batch_size
        assert pred_leye.shape[0] == batch_size
        assert pred_reye.shape[0] == batch_size
        assert body_pose_aa.shape[1] == 63
        
        pred_rotmat = pred_rotmat.view(batch_size, n_joints, 3, 3)
    
     # 分离全局旋转和身体姿态
        global_orient_mat = pred_rotmat[:, 0]  # [batch, 3, 3]
        global_orient_aa = rotation_matrix_to_angle_axis(global_orient_mat)  # [batch, 3]
        body_pose = pred_rotmat[:, 1:22]   # 只取1~21号关节 [batch, 21, 3, 3]
    
    # 关键修改：正确的轴角转换
        body_pose_flat = body_pose.reshape(-1, 3, 3)  # [batch*21, 3, 3]
        body_pose_aa = rotation_matrix_to_angle_axis(body_pose_flat)  # [batch*21, 3]
    
    # 计算每个姿态的轴角参数数量
        
        body_pose_aa = body_pose_aa.reshape(batch_size, 21*3)  # [batch, 63]
    
    # 验证尺寸
        assert body_pose_aa.shape[0] == batch_size
        assert body_pose_aa.shape[1] == 63
    

        # 保证所有参数 batch 维度一致
        for name, tensor in [
            ('pred_shape', pred_shape),
            ('body_pose_aa', body_pose_aa),
            ('global_orient_aa', global_orient_aa),
            ('pred_exp', pred_exp),
            ('pred_jaw', pred_jaw),
            ('pred_leye', pred_leye),
            ('pred_reye', pred_reye),
        ]:
            if tensor.shape[0] != batch_size:
                print(f'警告: {name} batch 维度为 {tensor.shape[0]}, 期望 {batch_size}')
                tensor = tensor.expand(batch_size, -1)

        if pred_exp.shape[0] != batch_size:
            print(f'警告: pred_exp batch 维度为 {pred_exp.shape[0]}, 期望 {batch_size}')
            pred_exp = pred_exp.expand(batch_size, -1)
        if pred_jaw.shape[0] != batch_size:
            print(f'警告: pred_jaw batch 维度为 {pred_jaw.shape[0]}, 期望 {batch_size}')
            pred_jaw = pred_jaw.expand(batch_size, -1)
        if pred_leye.shape[0] != batch_size:
            print(f'警告: pred_leye batch 维度为 {pred_leye.shape[0]}, 期望 {batch_size}')
            pred_leye = pred_leye.expand(batch_size, -1)
        if pred_reye.shape[0] != batch_size:
            print(f'警告: pred_reye batch 维度为 {pred_reye.shape[0]}, 期望 {batch_size}')
            pred_reye = pred_reye.expand(batch_size, -1)
        def ensure_batch(tensor, shape):
            if tensor.shape[0] != batch_size:
               return tensor.expand(batch_size, *tensor.shape[1:])
            return tensor

        pred_exp = ensure_batch(pred_exp, (batch_size, -1))
        pred_jaw = ensure_batch(pred_jaw, (batch_size, -1))
        pred_leye = ensure_batch(pred_leye, (batch_size, -1))
        pred_reye = ensure_batch(pred_reye, (batch_size, -1))



        # SMPLX前向计算 (关键修改)
        pred_output = self.smplx(
            betas=pred_shape,
            body_pose=body_pose_aa,  # 使用轴角表示
            global_orient=global_orient_aa,
            jaw_pose=pred_jaw,
            expression=pred_exp,
            leye_pose=pred_leye,  # 传入左眼姿态
            reye_pose=pred_reye,   # 传入右眼姿态
            left_hand_pose=pred_lhand,        # [B, 45]
            right_hand_pose=pred_rhand,       # [B, 45]
            pose2rot=False
        )

        # 后处理
        pred_vertices = pred_output.vertices
        pred_joints = pred_output.joints
        
        if J_regressor is not None:
            J_regressor_batch = J_regressor[None, :].expand(batch_size, -1, -1).to(x.device)
            pred_joints = torch.matmul(J_regressor_batch, pred_vertices)[:, H36M_TO_J14, :]

        pred_keypoints_2d = projection(pred_joints, pred_cam)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)  # 21×3=63

        return [{
            'theta': torch.cat([pred_cam, pose, pred_shape], 1),
            'verts': pred_vertices,
            'kp_2d': pred_keypoints_2d,
            'kp_3d': pred_joints,
            'rotmat': pred_rotmat
        }]