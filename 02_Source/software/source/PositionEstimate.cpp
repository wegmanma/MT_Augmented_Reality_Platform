#include <iostream>
#include <thread>

#include <PositionEstimate.hpp>

PositionEstimate::PositionEstimate()
{
    bmi160 = new BMI160(2, 0x69);
    quat_integrated[0] = 1; // Initial Quaternion: (1,0,0,0)
    tid = std::thread(&PositionEstimate::thrBMI160, this);
    tid.detach();
}

PositionEstimate::~PositionEstimate()
{
}

void PositionEstimate::get_gyro_data(vec3 gyro_data)
{

    gyro_data[0] = cumulated_rotation_xyz[0];
    gyro_data[1] = cumulated_rotation_xyz[1];
    gyro_data[2] = cumulated_rotation_xyz[2];
}

void PositionEstimate::get_gyro_matrix(mat4x4 gyro_matrix)
{
    mat4x4_dup(gyro_matrix, cumulated_mat);
}

void PositionEstimate::thrBMI160()
{

    std::chrono::steady_clock::time_point startTime;
    std::chrono::steady_clock::time_point endTime;
    std::chrono::steady_clock::duration timeSpan;

    if (bmi160->softReset() != BMI160_OK)
    {
        std::cout << "reset false" << std::endl;
    }
    if (bmi160->I2cInit(2, 0x69) != BMI160_OK)
    {
        std::cout << "init false" << std::endl;
        return;
    }

    int i = 0;
    int rslt;
    int16_t accelGyro[6] = {0};
    float theta;
    float inv_omega_len;
    quat temp_quat;

    vec3 norm_orientation;
    vec3 gyro_degree;
    mat4x4_identity(cumulated_mat);
    mat4x4_identity(temp_mat);
    rslt = bmi160->getAccelGyroData(accelGyro);
    if (rslt == 0)
    {
        for (i = 0; i < 3; i++)
        {
            // kill offset for gyro
            zero_rotation_xyz[i] = accelGyro[i] * 3.14 / 180.0;
        }
    }

    bmi160->setGyroFOC();
    while (1)
    {

        //get both accel and gyro data from bmi160
        //parameter accelGyro is the pointer to store the data
        startTime = std::chrono::steady_clock::now();
        rslt = bmi160->getAccelGyroData(accelGyro);
        endTime = std::chrono::steady_clock::now();
        std::chrono::steady_clock::duration timeSpan = endTime - startTime;
        double nseconds = double(timeSpan.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
        for (i = 0; i < 3; i++)
        {
            gyro_degree[i] = (accelGyro[i] * 0.061 * nseconds);//  / 180.0 * pi;
            cumulated_rotation_xyz[i] += (accelGyro[i] * 0.061 * nseconds);
            if (cumulated_rotation_xyz[i] >= 360.0f)
            {
                cumulated_rotation_xyz[i] -= 360.0f;
            }
            if (cumulated_rotation_xyz[i] <= 0.0f)
            {
                cumulated_rotation_xyz[i] += 360.0f;
            }
        }
        inv_omega_len = Q_rsqrt((float)(gyro_degree[0] * gyro_degree[0] + gyro_degree[1] * gyro_degree[1] + gyro_degree[2] * gyro_degree[2]));
        if (inv_omega_len != 0)
        {
            theta = nseconds / inv_omega_len;
            norm_orientation[0] = gyro_degree[0] * inv_omega_len;
            norm_orientation[1] = gyro_degree[1] * inv_omega_len;
            norm_orientation[2] = gyro_degree[2] * inv_omega_len;
            quat_gyro[0] = cos(theta / 2);
            quat_gyro[1] = norm_orientation[0] * sin(theta / 2);
            quat_gyro[2] = norm_orientation[1] * sin(theta / 2);
            quat_gyro[3] = norm_orientation[2] * sin(theta / 2);
            //====================================
            quat rot_quat;
            mat4x4 rotation;
            mat4x4 ident;
            mat4x4_identity(ident);
            mat4x4_rotate(rotation, ident, 0.0f, 0.0f, 1.0f, degreesToRadians(-90.0f));

            rot_quat[0] = 0.5 * sqrt(rotation[0][0] + rotation[1][1] + rotation[2][2] + 1);
            rot_quat[1] = 0.5 * sqrt(rotation[0][0] - rotation[1][1] - rotation[2][2] + 1);
            rot_quat[2] = 0.5 * sqrt(rotation[1][1] - rotation[2][2] - rotation[0][0] + 1);
            rot_quat[3] = 0.5 * sqrt(rotation[2][2] - rotation[0][0] - rotation[1][1] + 1);
            if ((rotation[2][1] - rotation[1][2]) < 0)
                rot_quat[1] *= -1;
            if ((rotation[0][2] - rotation[2][0]) < 0)
                rot_quat[2] *= -1;
            if ((rotation[1][0] - rotation[0][1]) < 0)
                rot_quat[3] *= -1;
            // std::cout << "Gyro Quaternion: (" << quat_gyro[0] << ", " << quat_gyro[1] << ", " << quat_gyro[2] << ", " << quat_gyro[3] << ")" << std::endl;
            // std::cout << "Rotation Quaternion: (" << rot_quat[0] << ", " << rot_quat[1] << ", " << rot_quat[2] << ", " << rot_quat[3] << ")" << std::endl;

            quat_matrix[0][0] = (quat_gyro[0] * quat_gyro[0]) + (quat_gyro[1] * quat_gyro[1]) - (quat_gyro[2] * quat_gyro[2]) - (quat_gyro[3] * quat_gyro[3]);
            quat_matrix[0][1] = 2.0 * (quat_gyro[1] * quat_gyro[2]) - 2.0 * (quat_gyro[0] * quat_gyro[3]);
            quat_matrix[0][2] = 2.0 * (quat_gyro[0] * quat_gyro[2]) + 2.0 * (quat_gyro[1] * quat_gyro[3]);
            quat_matrix[1][0] = 2.0 * (quat_gyro[0] * quat_gyro[3]) + 2.0 * (quat_gyro[1] * quat_gyro[2]);
            quat_matrix[1][1] = (quat_gyro[0] * quat_gyro[0]) - (quat_gyro[1] * quat_gyro[1]) + (quat_gyro[2] * quat_gyro[2]) - (quat_gyro[3] * quat_gyro[3]);
            quat_matrix[1][2] = 2.0 * (quat_gyro[2] * quat_gyro[3]) - 2.0 * (quat_gyro[0] * quat_gyro[1]);
            quat_matrix[2][0] = 2.0 * (quat_gyro[1] * quat_gyro[3]) - 2.0 * (quat_gyro[0] * quat_gyro[2]);
            quat_matrix[2][1] = 2.0 * (quat_gyro[0] * quat_gyro[1]) + 2.0 * (quat_gyro[2] * quat_gyro[3]);
            quat_matrix[2][2] = (quat_gyro[0] * quat_gyro[0]) - (quat_gyro[1] * quat_gyro[1]) - (quat_gyro[2] * quat_gyro[2]) + (quat_gyro[3] * quat_gyro[3]);
            mat4x4_mul(temp_mat,quat_matrix,cumulated_mat);
            mat4x4_dup(cumulated_mat,temp_mat);
            // std::cout << "quat_matrix = " << std::endl;
            // std::cout << "|" << quat_matrix[0][0] << " " << quat_matrix[1][0] << " " << quat_matrix[2][0] << " " << quat_matrix[3][0] << "|" << std::endl;
            // std::cout << "|" << quat_matrix[0][1] << " " << quat_matrix[1][1] << " " << quat_matrix[2][1] << " " << quat_matrix[3][1] << "|" << std::endl;
            // std::cout << "|" << quat_matrix[0][2] << " " << quat_matrix[1][2] << " " << quat_matrix[2][2] << " " << quat_matrix[3][2] << "|" << std::endl;
            // std::cout << "|" << quat_matrix[0][3] << " " << quat_matrix[1][3] << " " << quat_matrix[2][3] << " " << quat_matrix[3][3] << "|" << std::endl;
        }
    }
}

// Famous inverse square root algorithm from Quake III
float PositionEstimate::Q_rsqrt(float x)
{
    //return 1 / sqrt(x); // reference

    float halfx = 0.5f * x;
    float y = x;
    long i = *(long *)&y;      // evil floating point bit hack
    i = 0x5f3759df - (i >> 1); // what the fuck?
    y = *(float *)&i;
    y = y * (1.5f - (halfx * y * y)); // 1st iteration
    y = y * (1.5f - (halfx * y * y)); // 2nd iteration (can be removed for lower accuracy but more speed)
    return y;
}