# tensile_test.py

import os
import math
import random
import time
import numpy as np
import h5py
import csv
import pandas as pd

from isaacgym import gymapi, gymtorch, gymutil
import torch

# 최종에서는 생략하고 해보기
def cleanup_files():
    for file in ["ten_results_tensor_v2.hdf5", "ten_results.hdf5", "ten_particle_positions.hdf5", "strain_stress_summary.csv", "frame_indices.npy"]:
        if os.path.exists(file):
            os.remove(file)

def main():
    cleanup_files()

    gym = gymapi.acquire_gym()
    args = gymutil.parse_arguments(
        description="FEM Soft Body Tensile Test",
        custom_parameters=[{"name": "--headless", "action": "store_true", "help": "Run without viewer"}]
    )
    sim_params = setup_simulation(args)
    sim = create_simulation(gym, args, sim_params)
    gym.add_ground(sim, gymapi.PlaneParams())
    soft_asset = load_asset(gym, sim)
    env, soft_actor = create_environment(gym, sim, soft_asset)
    #viewer 조건부 활성화, 활성화X => python tensile_test.py --flex --headless
    viewer = None
    if not args.headless:
        viewer = create_viewer(gym, sim)
        gym.viewer_camera_look_at(viewer, None, gymapi.Vec3(-4.0, 2.8, -1.2), gymapi.Vec3(0.0, 1.4, 1.0))

    particle_state_tensor, initial_particle_state, tet_indices = initialize_tensors(gym, sim)
    top_particle_indices, bottom_particle_indices, center_y, window, initial_height = compute_geometry(initial_particle_state)
    top_idx, bottom_idx = select_gauge_particle_indices(initial_particle_state, center_y, window)
    start_time = gym.get_elapsed_time(sim)  #시뮬레이션 시간 측정(시작)

    run_sim_loop(gym, sim, viewer, sim_params, particle_state_tensor, initial_particle_state, tet_indices,
                 top_particle_indices, bottom_particle_indices, center_y, window, initial_height, top_idx, bottom_idx)
   
    end_time = gym.get_elapsed_time(sim)   #시뮬레이션 시간 측정(종료)
    print(f"[Isaac Gym 타이머 기준] 시뮬레이션 총 소요 시간: {end_time - start_time:.4f} 초")

    if viewer is not None:
        gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


def setup_simulation(args):
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 6000.0  
    sim_params.substeps = 2
    sim_params.flex.solver_type = 5
    sim_params.stress_visualization = True
    sim_params.flex.num_outer_iterations = 4
    sim_params.flex.num_inner_iterations = 20 
    sim_params.flex.relaxation = 0.8
    sim_params.flex.warm_start = 0.75
    sim_params.flex.deterministic_mode = True
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    #sim_params.stress_visualization_min = 0.0
    sim_params.stress_visualization_min = 1.e+2
    sim_params.stress_visualization_max = 1.e+4
    return sim_params

def create_simulation(gym, args, sim_params):
    sim = gym.create_sim(args.compute_device_id, args.compute_device_id, args.physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()
    return sim

# viewer 생성(GUI 시각화)
def create_viewer(gym, sim):
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        print("*** Failed to create viewer")
        quit()
    return viewer

# mesh 업로드
def load_asset(gym, sim):
    asset_root = "../../assets"
    soft_asset_file = "urdf/dogbone.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.thickness = 0.0  # 상호 침투 방지용임, 본 시뮬에서는 불필요
    asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
    return gym.load_asset(sim, asset_root, soft_asset_file, asset_options)

def create_environment(gym, sim, soft_asset):
    env = gym.create_env(sim, gymapi.Vec3(-3, 0, -3), gymapi.Vec3(3, 3, 3), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0.0, 1.0, 0.0)
    actor = gym.create_actor(env, soft_asset, pose, "soft", 0, 1)
    return env, actor

# 데이터 계산 : particle의 텐서, 초기 상태 리턴
def initialize_tensors(gym, sim):
    particle_state_tensor = gymtorch.wrap_tensor(gym.acquire_particle_state_tensor(sim))
    gym.refresh_particle_state_tensor(sim)
    initial_particle_state = particle_state_tensor.clone()
    tet_indices, _ = gym.get_sim_tetrahedra(sim)
    return particle_state_tensor, initial_particle_state, tet_indices

# 데이터 계산 : 입자 위치 트랙킹 및 범위 선택 
def compute_geometry(initial_particle_state):
    max_y = torch.max(initial_particle_state[:, 1]).item()
    min_y = torch.min(initial_particle_state[:, 1]).item()
    center_y = (max_y + min_y) / 2.0
    window = 0.025
    top_particle_indices = torch.where(initial_particle_state[:, 1] >= (max_y - 0.024))[0]
    bottom_particle_indices = torch.where(initial_particle_state[:, 1] <= (min_y + 0.024))[0]
    initial_height = max_y - min_y
    return top_particle_indices, bottom_particle_indices, center_y, window, initial_height

# strain 측정용 입자 인덱스 선택
def select_gauge_particle_indices(initial_particle_state, center_y, window):
    y_coords = initial_particle_state[:, 1]
    upper_indices = torch.where(y_coords >= center_y + window)[0]
    lower_indices = torch.where(y_coords <= center_y - window)[0]
    top_idx = upper_indices[torch.argmax(y_coords[upper_indices])].item()
    bottom_idx = lower_indices[torch.argmin(y_coords[lower_indices])].item()
    return top_idx, bottom_idx

# 텐서의 (선형)변형률 계산
def compute_tet_strain_tensor(p0, p1, p2, p3, u0, u1, u2, u3):
    Dm = np.column_stack((p1 - p0, p2 - p0, p3 - p0))
    Du = np.column_stack((u1 - u0, u2 - u0, u3 - u0))
    F = Du @ np.linalg.inv(Dm)  # Dm의 역행렬
    return 0.5 * (F + F.T)      # strain tensor

# von mises stress 계산
def extract_stress_strain_with_gauge(gym, sim, initial_positions, particle_state_tensor, selected_tet_indices):
    tet_indices, stresses = gym.get_sim_tetrahedra(sim)
    stress_von_mises = []

    for tet_idx in selected_tet_indices:
        base_idx = tet_idx * 4
        ids = tet_indices[base_idx:base_idx + 4]
        init_pos = np.array([initial_positions[i][:3] for i in ids])
        curr_pos = np.array([particle_state_tensor[i, :3].cpu().numpy() for i in ids])
        displ = curr_pos - init_pos

        stress_tensor = np.array([
            [stresses[tet_idx].x.x, stresses[tet_idx].y.x, stresses[tet_idx].z.x],
            [stresses[tet_idx].x.y, stresses[tet_idx].y.y, stresses[tet_idx].z.y],
            [stresses[tet_idx].x.z, stresses[tet_idx].y.z, stresses[tet_idx].z.z]
        ])
        vm_stress = np.sqrt(0.5 * ((stress_tensor[0, 0] - stress_tensor[1, 1]) ** 2 +
                                   (stress_tensor[1, 1] - stress_tensor[2, 2]) ** 2 +
                                   (stress_tensor[2, 2] - stress_tensor[0, 0]) ** 2 +
                                   6 * (stress_tensor[1, 2] ** 2 + stress_tensor[2, 0] ** 2 + stress_tensor[0, 1] ** 2)))

        stress_von_mises.append(vm_stress)

    return np.array(stress_von_mises)

def run_sim_loop(gym, sim, viewer, sim_params, particle_state_tensor, initial_particle_state, tet_indices,
                 top_particle_indices, bottom_particle_indices, center_y, window, initial_height, top_idx, bottom_idx):

    total_time = 10.0     # 총 시뮬레이션 시간 (초)
    num_steps = int(total_time / sim_params.dt)   # 전체 프레임 수 계산 = 총 시간 / 타임스텝
    displacement_per_frame = 0.005 / num_steps    # 프레임당 이동 거리 = 총 0.005m를 나눠서 한 프레임씩 이동
    target_save_frames = 100                      # 저장할 타겟 프레임 수를 100개로 설정
    SAVE_EVERY = max(1, num_steps // target_save_frames)  # 저장 주기 계산

    # 초기 설정
    stress_summary_rows = []
    saved_frames = []
    initial_positions = initial_particle_state.cpu().numpy()  # 초기 입자 위치를 numpy 배열로 변환
    tet_indices = np.array(tet_indices)
    if tet_indices.ndim == 1:                                 # tet 요소 인덱스를 numpy 배열로 변환
        tet_indices = tet_indices.reshape(-1, 4)

    # 중심 위치 계산 -> 관심 요소 인덱스 필터링
    initial_centroids = np.mean(initial_positions[tet_indices], axis=1)
    # 중심 영역(center_y ± window)에 포함되는 요소만 선택
    mask = (initial_centroids[:, 1] >= center_y - window) & (initial_centroids[:, 1] <= center_y + window)
    selected_tet_indices = np.where(mask)[0]
    # 중심 영역 중 y좌표가 가장 높은, 가장 낮은 tetra 요소 선택
    top_centroid_idx = selected_tet_indices[np.argmax(initial_centroids[mask][:, 1])]
    bottom_centroid_idx = selected_tet_indices[np.argmin(initial_centroids[mask][:, 1])]
    top_y0 = initial_centroids[top_centroid_idx][1]
    bottom_y0 = initial_centroids[bottom_centroid_idx][1]
    initial_delta_y = (top_y0 - bottom_y0)

    # HDF5 저장소 초기화
    stress_h5 = h5py.File("selected_von_mises_stress.hdf5", "w")
    strain_h5 = h5py.File("selected_von_mises_strain.hdf5", "w")

    frame = 0
    while (viewer is None or not gym.query_viewer_has_closed(viewer)) and frame < num_steps:
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # 저장할 프레임인지 판단 (처음, 마지막, SAVE_EVERY 간격)
        save_now = (frame % SAVE_EVERY == 0 or frame == 0 or frame == num_steps - 1)
        if save_now:
            saved_frames.append(frame)
            particle_positions_frame = particle_state_tensor[:, :3].cpu().numpy()

            # 선택 요소만 stress/strain 계산
            stress_sel = extract_stress_strain_with_gauge(
                gym, sim, initial_positions, particle_state_tensor, selected_tet_indices)

            # HDF5 저장
            stress_h5.create_dataset(f"frame_{frame}", data=stress_sel)

            # gauge strain 계산
            centroids = np.mean(particle_positions_frame[tet_indices], axis=1)
            top_y = centroids[top_centroid_idx][1]
            bottom_y = centroids[bottom_centroid_idx][1]
            current_delta_y = (top_y - bottom_y)
            gauge_strain = round((current_delta_y - initial_delta_y) / (initial_delta_y + 1e-8), 6)

            # summary row 저장
            stress_summary_rows.append([
                frame,
                np.mean(stress_sel),
                np.max(stress_sel),
                gauge_strain
            ])

        # # 하단 입자의 속도를 0으로 고정 (움직이지 않게)
        # particle_state_tensor[bottom_particle_indices, 3:6] = 0.0
        # # 상단 입자를 위로 이동시켜 인장 유도
        # particle_state_tensor[top_particle_indices, 1] += displacement_per_frame
        # # 상단 입자의 속도도 0으로 고정 (이동만 하고 가속도는 없게)
        # particle_state_tensor[top_particle_indices, 3:6] = 0.0  # 상단 속도도 반드시 0으로 고정
        
        gym.set_particle_state_tensor(sim, gymtorch.unwrap_tensor(particle_state_tensor))
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, False)
        frame += 1

    stress_h5.close()
    strain_h5.close()
    np.save("frame_indices.npy", np.array(saved_frames))

    with open("strain_stress_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Frame", "Stress [Pa]", "v_Stress_Max", "Strain [m/m]"])
        writer.writerows(stress_summary_rows)

    print("strain_stress_summary.csv 저장 완료")


# 데이터의 빠른 파악을 위해 임시 작성해둠
def count_saved_frames(csv_path="strain_stress_summary.csv"):
    try:
        df = pd.read_csv(csv_path)
        num_frames = len(df)
        print(f"저장된 총 프레임 수: {num_frames}")
        print(f"프레임 인덱스 (처음 5개): {df['Frame'].head().tolist()}")
        print(f"프레임 인덱스 (마지막 5개): {df['Frame'].tail().tolist()}")
        print(f"y_strain (마지막 5개): {df['Strain [m/m]'].tail().tolist()}")
    except FileNotFoundError:
        print(f"{csv_path} 파일이 존재하지 않습니다.")

if __name__ == "__main__":
    main()
    count_saved_frames()
