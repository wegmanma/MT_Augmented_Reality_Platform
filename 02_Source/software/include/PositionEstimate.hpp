#pragma once
#include "BMI160.hpp"
#include <thread>
#include "linmath.h"

class PositionEstimate  {

public:

PositionEstimate();
~PositionEstimate();

void get_gyro_data(vec3 gyro_data);
void get_gyro_matrix(mat4x4 gyro_matrix);

private:

    BMI160 *bmi160;

    void thrBMI160();

    quat quat_integrated{}; // q(t), q_0 = (1,0,0,0)
    quat quat_gyro{}; // instantaneous rotation

    mat4x4 quat_matrix{};
    mat4x4 temp_mat;
    mat4x4 cumulated_mat;

    vec3 zero_rotation_xyz = { 0.0, 0.0, 0.0}; 

    vec3 cumulated_rotation_xyz = { 0.0, 0.0, 0.0}; 
    vec4 cumulated_translation_xyz = { 0.0, 0.0, 0.0, 1.0};
    const float pi = 3.14159265;
    std::thread tid;

    // famous inverse square root x from Quake III

    float Q_rsqrt(float x);
};