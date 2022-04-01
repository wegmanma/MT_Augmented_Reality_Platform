#pragma once
#include "BMI160.hpp"
#include <mutex>
#include <thread>
#include "linmath.h"
#include "TCPFrameCapture.hpp"


class PositionEstimate  {

public:

PositionEstimate(TCPFrameCapture *tcpCapture_p);
~PositionEstimate();

void get_gyro_matrix(mat4x4 &gyro_matrix, mat4x4 &translation);

private:

    BMI160 *bmi160;
    TCPFrameCapture *tcpCapture;

    void thrBMI160();

    std::mutex mtx;

    int cycle_count{};

    vec3 gyro_rad_per_s{};
    vec4 accel_m_per_sq_s{};
    float vec_len{};

    int16_t accelGyro[9] = {0};
    int16_t accelGyroLast[9] = {0};
    uint32_t time, time_last;
    double nseconds;
    int meas_cnt;

    vec17 x = {0};
    vec17 K = {0};
    vec17 x_apriori = {0};
    mat17x17 F = {0};
    mat17x17 P_apriori = {0};
    mat17x17 P_aposteriori = {0};

    // rotation quaternions
    quat quat_integrated{}; // q(t), q_0 = (1,0,0,0)
    quat quat_gyro{};       // instantaneous rotation measured by gyro
    quat quat_correction{}; // correction rotation by accelerometer ("up direction")
    

    // orientation quaternions
    quat initial_orientation{}; // (a,b,c) measured at start (up direction)
    quat updated_orientation{}; // (a,b,c) current quat_integrated applied to initial_orientation;
    quat measured_orientation{}; // (a,b,c) up direction measured whenever device is still, corrects "updated_orientation"
    // position estimation
    quat delta_vector_abc{}; // acceleration without G in ABC-direction
    quat delta_vector_xyz{}; // true accelerations in XYZ-direction
    quat delta_vector_xyz_moving{}; // used for filtering.
    quat delta_tof_vector_moving{};

    quat kalman_rotation{};
    quat kalman_translation{};
    quat kalman_translation_from_velocity{};

    mat4x4 quat_matrix{}; // final matrix to hand over to Vulkan
    quat tof_translation_abc{};
    quat tof_translation_xyz{};
    quat quat_tof_integrated{}; // temp matrix used for estimating how ToF camera motion develops over time

    vec3 zero_rotation_abc = { 0.0, 0.0, 0.0}; 
    vec3 accel_gain_correct = { 0.0, 0.0, 0.0};
    vec3 zero_translation_abc = { 0.0, 0.0, 0.0}; 

    const float pi = 3.14159265;
    std::thread tid;

    // famous inverse square root x from Quake III

    float Q_rsqrt(float x);

void quat_from_angle_axis(quat R, float radians, vec3 axis);

void quat_RotationBetweenVectors(quat R, vec3 S, vec3 D);
};