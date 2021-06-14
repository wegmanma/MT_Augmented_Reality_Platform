#pragma once
#include "BMI160.hpp"
#include <thread>
#include "linmath.h"

class PositionEstimate  {

public:

PositionEstimate();
~PositionEstimate();

void get_gyro_data(vec3 gyro_data);

private:

    BMI160 *bmi160;

    void thrBMI160();

    vec3 zero_rotation_xyz = { 0.0, 0.0, 0.0}; 

    vec3 cumulated_rotation_xyz = { 0.0, 0.0, 0.0}; 
    vec3 cumulated_translation_xyz = { 0.0, 0.0, 0.0};
    std::thread tid;
};