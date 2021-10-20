#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <cstring>
#include <iomanip>
#include <pthread.h>
#include <PositionEstimate.hpp>

// #define SAVE_DATA

#define CONST_G 9.81

#define NO_ROTATION_TRESH 0.02
#define NO_MOVEMENT_MAX_COUNT 5
#define ACC_FILTER_DEPTH 0.1 // 1 = no filter, 0 = no update

#define GYRO_RESOLUTION 0.061
#define ACCEL_RESOLUTION 1670.13 * 4

#define SENS_TIME_RESOLUTION 0.000039

#define OFFSET_MEAS_TIME 1000
#define CAL_GYRO_X 0.0056189002  //2.381310
#define CAL_GYRO_Y -0.0020643455 //-2.783860
#define CAL_GYRO_Z 0.0082736398  //6.784730
#define CAL_ACC_X -153.7980625   // -65.8768
#define CAL_ACC_Y 75.7320625     // -90.7988
#define CAL_ACC_Z 30.81325       // 41.24667
// orientational corrections
#define FACE_UP_CORR 0.9821081568    //0.00163
#define FACE_DOWN_CORR 0.9966960825  //0.018828
#define FLIP_LEFT_CORR 0.9882184034  //0.011186
#define FLIP_RIGHT_CORR 0.9936513847 //0.006859
#define FLIP_FRONT_CORR 0.9944487206 //0.001764
#define FLIP_BACK_CORR 1.0004113867  //-0.006346

bool GyroDataEqual(int16_t *old_data, int16_t *new_data)
{
    for (int i = 0; i < 6; i++)
        if (old_data[i] != new_data[i])
            return false;
    return true;
}

void quat_rotate(quat result, quat rotation, quat vector)
{
    quat conj_rot;
    quat temp;
    quat_conj(conj_rot, rotation);
    quat_mul(temp, conj_rot, vector);
    quat_mul(result, temp, rotation);
}

void PositionEstimate::quat_from_angle_axis(quat R, float radians, vec3 axis)
{
    vec3 axis_norm;
    vec3_norm(axis_norm, axis);
    R[3] = cos(radians / 2);
    R[0] = axis_norm[0] * sin(radians / 2);
    R[1] = axis_norm[1] * sin(radians / 2);
    R[2] = axis_norm[2] * sin(radians / 2);
}

void PositionEstimate::quat_RotationBetweenVectors(quat R, vec3 S, vec3 D)
{
    vec3 start, dest;
    vec3_norm(start, S);
    vec3_norm(dest, D);

    float cosTheta = vec3_mul_inner(start, dest);
    vec3 rotationAxis;

    if (cosTheta < -1 + 0.001f)
    {
        // special case when vectors in opposite directions:
        // there is no "ideal" rotation axis
        // So guess one; any will do as long as it's perpendicular to start
        vec3 ref = {0.0f, 0.0f, 1.0f};
        vec3_mul_cross(rotationAxis, ref, start);
        if (vec3_len(rotationAxis) < 0.01)
        {
            // bad luck, they were parallel, try again!
            ref[0] = 1.0f;
            ref[2] = 0.0f;
            vec3_mul_cross(rotationAxis, ref, start);
        }
        vec3_norm(rotationAxis, rotationAxis);
        quat_from_angle_axis(R, pi, rotationAxis);
        return;
    }

    vec3_mul_cross(rotationAxis, start, dest);

    float s = sqrt((1 + cosTheta) * 2);
    float invs = 1 / s;

    R[3] = s * 0.5f;
    R[0] = rotationAxis[0] * invs;
    R[1] = rotationAxis[1] * invs;
    R[2] = rotationAxis[2] * invs;
    return;
}

bool check_around_zero(float val, float tresh)
{
    if ((val < tresh) && (val > (-1.0) * tresh))
        return true;
    else
        return false;
}

bool vec3_check_around_zero(vec3 val, float tresh)
{
    for (int i = 0; i < 3; i++)
    {
        if (!check_around_zero(val[i], tresh))
            return false;
    }
    return true;
}

PositionEstimate::PositionEstimate(TCPFrameCapture *tcpCapture_p)
{
    bmi160 = new BMI160(2, 0x69);
    tcpCapture = tcpCapture_p;
    tid = std::thread(&PositionEstimate::thrBMI160, this);
    sched_param sch;
    int policy;
    pthread_getschedparam(tid.native_handle(), &policy, &sch);
    sch.sched_priority = 20;
    if (pthread_setschedparam(tid.native_handle(), SCHED_FIFO, &sch))
    {
        std::cout << "Failed to setschedparam: " << std::strerror(errno) << '\n';
    }
    tid.detach();
}

PositionEstimate::~PositionEstimate()
{
}

void PositionEstimate::get_gyro_matrix(mat4x4 gyro_matrix)
{
    mat4x4 ToFrotation;
    vec4 ToFtranslation;
    quat ToFQuaterion;
    bool newdata;
    int ToFmtx = tcpCapture->lockMutex();
    newdata = tcpCapture->getRotationTranslation(ToFmtx, ToFrotation, ToFtranslation);
    tcpCapture->unlockMutex(ToFmtx);
    if (newdata == true)
    {
        int meas_cnt_i;
        mtx.lock();
        vec4 accel_m_per_sq_s_i;
        meas_cnt_i = meas_cnt;
        accel_m_per_sq_s_i[1] = accel_m_per_sq_s[0] / (meas_cnt);
        accel_m_per_sq_s_i[2] = accel_m_per_sq_s[1] / (meas_cnt);
        accel_m_per_sq_s_i[0] = -accel_m_per_sq_s[2] / (meas_cnt);
        accel_m_per_sq_s_i[3] = accel_m_per_sq_s[3] / (meas_cnt);
        // std::cout << "get_gyro_matrix: " << nseconds << std::endl;
        double nseconds_i = nseconds;
        vec3 gyro_rad_per_s_i;
        double theta;
        gyro_rad_per_s_i[1] = -gyro_rad_per_s[0] / (meas_cnt);
        gyro_rad_per_s_i[2] = gyro_rad_per_s[1] / (meas_cnt);
        gyro_rad_per_s_i[0] = gyro_rad_per_s[2] / (meas_cnt);
        accel_m_per_sq_s[0] = 0.0f;
        accel_m_per_sq_s[1] = 0.0f;
        accel_m_per_sq_s[2] = 0.0f;
        accel_m_per_sq_s[3] = 0.0f;
        gyro_rad_per_s[0] = 0.0f;
        gyro_rad_per_s[1] = 0.0f;
        gyro_rad_per_s[2] = 0.0f;
        meas_cnt = 0;
        nseconds = 0.0f;
        mtx.unlock();
        // std::cout << meas_cnt_i << ";" << nseconds_i << ";" <<gyro_rad_per_s_i[0] << ";" << gyro_rad_per_s_i[1] << ";" << gyro_rad_per_s_i[2];
        // std::cout << ";" << accel_m_per_sq_s_i[0] << ";" << accel_m_per_sq_s_i[1] << ";" << accel_m_per_sq_s_i[2] << ";" << accel_m_per_sq_s_i[3] << std::endl;

        if (meas_cnt_i > 0)
        {
            // Calculate vector length
            for (int i = 0; i < 3; i++)
            {
                vec_len += accel_m_per_sq_s_i[i] * accel_m_per_sq_s_i[i];
            }
            vec_len = sqrt(vec_len);
            float inv_omega_len;
            // get rotation quaternion from gyro data
            inv_omega_len = Q_rsqrt((float)(gyro_rad_per_s_i[0] * gyro_rad_per_s_i[0] + gyro_rad_per_s_i[1] * gyro_rad_per_s_i[1] + gyro_rad_per_s_i[2] * gyro_rad_per_s_i[2]));
            if ((inv_omega_len != 0.0) && (gyro_rad_per_s_i[0] != 0.0f) && (gyro_rad_per_s_i[1] != 0.0f) && (gyro_rad_per_s_i[2] != 0.0f))
            {
                theta = nseconds_i / inv_omega_len;
                quat_from_angle_axis(quat_gyro, -theta, gyro_rad_per_s_i);
            }
            else // no rotation, identity quaternion "does nothing"
            {
                quat_identity(quat_gyro);
            }

            quat_from_mat4x4(ToFQuaterion, ToFrotation);

            if (ToFrotation[0][0] > 0.75)
            {
                quat temp;
                quat_mul(temp, quat_tof_integrated, ToFQuaterion);
                quat_tof_integrated[0] = temp[0];
                quat_tof_integrated[1] = temp[1];
                quat_tof_integrated[2] = temp[2];
                quat_tof_integrated[3] = temp[3];
            }
            // print_quat("quat_tof_integrated",quat_tof_integrated);
            // print_quat("quat_integrated",quat_integrated);
            // print_mat4x4("ToFrotation",ToFrotation);

            // update rotation quaternion
            {
                quat temp{};
                quat_mul(temp, quat_gyro, quat_integrated);
                quat_integrated[0] = temp[0];
                quat_integrated[1] = temp[1];
                quat_integrated[2] = temp[2];
                quat_integrated[3] = temp[3];
            }
            // print_quat("ToFQuaterion", ToFQuaterion, true, true);
            // print_quat("quat_gyro", quat_gyro, true);
            print_vec4("tofTranslation", ToFtranslation, true, true);
            // compute current measured orientation - might be unstable, only apply when stable
            for (int i = 0; i < 3; i++)
            {
                measured_orientation[i] = accel_m_per_sq_s_i[i] / vec_len;
            }
            // compute orientation as seen by quat_integrated
            // print_quat("initial_orientation", initial_orientation);
            // print_quat("measured_orientation", measured_orientation);
            {
                quat quat_integrated_conj{};
                quat temp{};
                quat_conj(quat_integrated_conj, quat_integrated);
                quat_mul(temp, quat_integrated, initial_orientation);
                quat_mul(updated_orientation, temp, quat_integrated_conj);
            }
            // compute correction rotation
            {
                quat_RotationBetweenVectors(quat_correction, updated_orientation, measured_orientation);
            }
            // compute velocity without gravity
            quat delta_vector_abc{};
            {

                quat quat_integrated_conj{};
                quat temp{};
                for (int i = 0; i < 3; i++)
                {
                    delta_vector_abc[i] = accel_m_per_sq_s_i[i] - updated_orientation[i] * CONST_G;
                }
                quat_conj(quat_integrated_conj, quat_integrated);
                quat_mul(temp, quat_integrated_conj, delta_vector_abc);
                quat_mul(delta_vector_xyz, temp, quat_integrated);
            }
            print_quat("delta_vector_abc", delta_vector_abc, true);
            // if no movement and no rotation: Use accelerometer to find G

            if ((vec_len >= 9.77) && (vec_len <= 9.86) && vec3_check_around_zero(gyro_rad_per_s, NO_ROTATION_TRESH))
            {
                // std::cout << "Stable: \U00002705 (no motion)" << std::endl;
                for (int i = 0; i < 3; i++)
                {
                    delta_vector_xyz_moving[i] = (1 - ACC_FILTER_DEPTH) * delta_vector_xyz_moving[i] + ACC_FILTER_DEPTH * delta_vector_xyz[i];
                }
                // correct rotation quaternion
                {
                    quat temp{};
                    quat_mul(temp, quat_correction, quat_integrated);
                    quat_integrated[0] = temp[0];
                    quat_integrated[1] = temp[1];
                    quat_integrated[2] = temp[2];
                    quat_integrated[3] = temp[3];
                }
            }
            else
            {
                // std::cout << "Stable: \U0000274C (motion detected)" << std::endl;
                // print_quat("quat integrated", quat_integrated);
                for (int i = 0; i < 3; i++)
                {
                    // delta_vector_xyz_moving[i] = (1-ACC_FILTER_DEPTH)*delta_vector_xyz_moving[i]+0.0;
                }
            }
            for (int i = 0; i < 3; i++)
            {
                delta_vector_xyz[i] = delta_vector_xyz[i] - delta_vector_xyz_moving[i];
            }
            // calculate velocity from acceleration
            for (int i = 0; i < 3; i++)
            {
                velocity_xyz[i] += delta_vector_xyz[i] * nseconds_i;
            }

            if ((vec_len >= 9.77) && (vec_len <= 9.86) && vec3_check_around_zero(gyro_rad_per_s, NO_ROTATION_TRESH))
            {
                // quat_identity(velocity_xyz);
            }
            // calculate position from velocity
            for (int i = 0; i < 3; i++)
            {
                position_xyz[i] += velocity_xyz[i] * nseconds_i;
            }
            std::cout << std::fixed << std::setprecision(15);

            {
                // update the matrix
                mat4x4_from_quat(quat_matrix, quat_integrated);
            }
            for (int i = 0; i < 9; i++)
            {
                accelGyroLast[i] = accelGyro[i];
            }
        }
    }
    //print_mat4x4("ToFrotation", ToFrotation);
    //print_mat4x4("quat_matrix",quat_matrix);

    mat4x4_dup(gyro_matrix, quat_matrix);
}

void PositionEstimate::thrBMI160()
{

    if (bmi160->softReset() != BMI160_OK)
    {
        std::cout << "reset false" << std::endl;
    }
    if (bmi160->I2cInit(9, 0x69) != BMI160_OK)
    {
        std::cout << "init false" << std::endl;
        return;
    }

    quat_integrated[3] = 1; // Initial Quaternion: (1|0,0,0)
    quat_correction[3] = 1; // Initial Quaternion: (1|0,0,0)

    // raw AccelGyro-Data

    //Quaternion Calculation Variables

    // general positioning calculation

    quat_tof_integrated[0] = 0.0f;
    quat_tof_integrated[1] = 0.0f;
    quat_tof_integrated[2] = 0.0f;
    quat_tof_integrated[3] = 1.0f;

    accel_m_per_sq_s[3] = 1.0f;

    zero_rotation_abc[0] = CAL_GYRO_X;
    zero_rotation_abc[1] = CAL_GYRO_Y;
    zero_rotation_abc[2] = CAL_GYRO_Z;
    zero_translation_abc[0] = CAL_ACC_X;
    zero_translation_abc[1] = CAL_ACC_Y;
    zero_translation_abc[2] = CAL_ACC_Z;

    bmi160->getAccelGyroData(accelGyro);
    for (int i = 0; i < 9; i++)
    {
        accelGyroLast[i] = accelGyro[i];
    }
    for (int i = 0; i < 3; i++)
    {
        gyro_rad_per_s[i] = (((accelGyro[i]) * GYRO_RESOLUTION) / 180.0 * pi) - zero_rotation_abc[i];
        accel_m_per_sq_s[i] = ((accelGyro[i + 3] - zero_translation_abc[i]) / ACCEL_RESOLUTION);
    }

    // gain correction
    if (accel_m_per_sq_s[0] <= 0)
        accel_m_per_sq_s[0] *= FACE_DOWN_CORR; // x_neg
    else
        accel_m_per_sq_s[0] *= FACE_UP_CORR; // x_pos

    if (accel_m_per_sq_s[1] <= 0)
        accel_m_per_sq_s[1] *= FLIP_RIGHT_CORR; // y_neg
    else
        accel_m_per_sq_s[1] *= FLIP_LEFT_CORR; // y_pos

    if (accel_m_per_sq_s[2] <= 0)
        accel_m_per_sq_s[2] *= FLIP_BACK_CORR; // z_neg
    else
        accel_m_per_sq_s[2] *= FLIP_FRONT_CORR; // z_pos
    // Calculate vector length

    vec_len = 0.0f;
    for (int i = 0; i < 3; i++)
    {
        vec_len += accel_m_per_sq_s[i] * accel_m_per_sq_s[i];
    }
    vec_len = sqrt(vec_len);
    // normalized up-direction at start

    initial_orientation[1] = accel_m_per_sq_s[0] / vec_len;
    initial_orientation[2] = accel_m_per_sq_s[1] / vec_len;
    initial_orientation[0] = -accel_m_per_sq_s[2] / vec_len;

    int file = bmi160->I2CGetDataOpenDevice();
    double nseconds_i;
    vec3 gyro_i;
    vec4 accel_i;
    while (1)
    {

        //get both accel and gyro data from bmi160
        //parameter accelGyro is the pointer to store the data
        bmi160->getAccelGyroDataFast(file, accelGyro);
        // get sensor timestamp
        time = (uint32_t)((accelGyro[6] << 16) || (accelGyro[7]));
        // std::cout << "beginning of loop: " << nseconds_i << std::endl;
        if (accelGyroLast[6] > accelGyro[6])
            nseconds_i = nseconds_i + (double)(accelGyro[6] - accelGyroLast[6] + 65536) * SENS_TIME_RESOLUTION;
        else
            nseconds_i = nseconds_i + (double)(accelGyro[6] - accelGyroLast[6]) * SENS_TIME_RESOLUTION;

        if (GyroDataEqual(accelGyro, accelGyroLast))
        {
            // std::cout << "continuing" << std::endl;
            continue;
        }
        // extract all raw measurements with offset-correction
        vec_len = 0.0f;

        for (int i = 0; i < 3; i++)
        {
            gyro_i[i] = (((accelGyro[i]) * GYRO_RESOLUTION) / 180.0 * pi) - zero_rotation_abc[i];
            accel_i[i] = ((accelGyro[i + 3] - zero_translation_abc[i]) / ACCEL_RESOLUTION);
        }
        accel_i[3] = 1.0;

        // gain correction
        if (accel_i[0] <= 0)
            accel_i[0] *= FLIP_BACK_CORR; // x_neg
        else
            accel_i[0] *= FLIP_FRONT_CORR; // x_pos

        if (accel_i[1] <= 0)
            accel_i[1] *= FLIP_RIGHT_CORR; // y_neg
        else
            accel_i[1] *= FLIP_LEFT_CORR; // y_pos

        if (accel_i[2] <= 0)
            accel_i[2] *= FACE_DOWN_CORR; // z_neg
        else
            accel_i[2] *= FACE_UP_CORR; // z_pos
        mtx.lock();
        meas_cnt++;

        nseconds = nseconds_i;
        // std::cout << "main_thread, t= " << nseconds_i << "gyro = " << gyro_i[0] << "; " << gyro_i[1] << "; " << gyro_i[2] << std::endl;
        nseconds_i = 0.0;
        gyro_rad_per_s[0] += gyro_i[0];
        gyro_rad_per_s[1] += gyro_i[1];
        gyro_rad_per_s[2] += gyro_i[2];
        accel_m_per_sq_s[0] += accel_i[0];
        accel_m_per_sq_s[1] += accel_i[1];
        accel_m_per_sq_s[2] += accel_i[2];
        accel_m_per_sq_s[3] += accel_i[3];
        mtx.unlock();
    }
    bmi160->I2CGetDataCloseDevice(file);
}

// Famous inverse square root algorithm from Quake III
float PositionEstimate::Q_rsqrt(float x)
{
    return 1 / sqrt(x); // reference

    float halfx = 0.5f * x;
    float y = x;
    long i = *(long *)&y;      // evil floating point bit hack
    i = 0x5f3759df - (i >> 1); // what the fuck?
    y = *(float *)&i;
    y = y * (1.5f - (halfx * y * y)); // 1st iteration
    y = y * (1.5f - (halfx * y * y)); // 2nd iteration (can be removed for lower accuracy but more speed)
    return y;
}
