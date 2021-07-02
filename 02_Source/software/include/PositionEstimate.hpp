#pragma once
#include "BMI160.hpp"
#include <mutex>
#include <thread>
#include "linmath.h"

class PositionEstimate  {

public:

PositionEstimate();
~PositionEstimate();

void get_gyro_matrix(mat4x4 gyro_matrix);

private:

    BMI160 *bmi160;

    void thrBMI160();

    std::mutex mtx;

    // orientation quaternions
    quat initial_orientation{}; // (a,b,c) measured at start (up direction)
    quat updated_orientation{}; // (a,b,c) current quat_integrated applied to initial_orientation;
    quat measured_orientation{}; // (a,b,c) up direction measured whenever device is still, corrects "updated_orientation"
    // position estimation
    quat delta_vector_xyz{}; // true accelerations
    quat delta_vector_xyz_moving{}; // used for filtering.
    vec4 position_xyz{}; // (x,y,z) Physical position change from start in m
    vec4 velocity_xyz{}; // (x,y,z) Current velocity in m/s 
    // (x,y,z) acceleration is instantaneous and therefore kept within function
    // (a,b,c) acceleration is instantaneous and therefore kept within function

    mat4x4 quat_matrix{}; // final matrix to hand over to Vulkan

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