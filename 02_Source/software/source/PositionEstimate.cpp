#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <iomanip>
#include <PositionEstimate.hpp>

#define CONST_G 9.81

#define NO_ROTATION_TRESH 0.02
#define NO_MOVEMENT_MAX_COUNT 5
#define ACC_FILTER_DEPTH 0.1 // 1 = no filter, 0 = no update

#define GYRO_RESOLUTION 0.061
#define ACCEL_RESOLUTION 1670.13 * 4

#define SENS_TIME_RESOLUTION 0.000039

#define OFFSET_MEAS_TIME 1000
#define CAL_GYRO_X 2.381310
#define CAL_GYRO_Y -2.783860
#define CAL_GYRO_Z 6.784730
#define CAL_ACC_X -153.7980625 // -65.8768
#define CAL_ACC_Y -105.7320625 // -90.7988
#define CAL_ACC_Z 30.81325     // 41.24667
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

void print_quat(std::string name, quat e)
{
    std::cout << std::fixed << std::setprecision(5);
    std::cout << std::setw(20) << name << " = (" << std::setw(8) << e[3] << " | " << std::setw(8) << e[0] << ", " << std::setw(8) << e[1] << ", " << std::setw(8) << e[2] << ")" << std::endl;
    std::cout << std::defaultfloat << std::setprecision(6);
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

PositionEstimate::PositionEstimate()
{
    bmi160 = new BMI160(2, 0x69);

    tid = std::thread(&PositionEstimate::thrBMI160, this);
    tid.detach();
}

PositionEstimate::~PositionEstimate()
{
}

void PositionEstimate::get_gyro_matrix(mat4x4 gyro_matrix)
{
    mtx.lock();
    mat4x4_dup(gyro_matrix, quat_matrix);
    mtx.unlock();
}

void PositionEstimate::thrBMI160()
{

    if (bmi160->softReset() != BMI160_OK)
    {
        std::cout << "reset false" << std::endl;
    }
    if (bmi160->I2cInit(8, 0x69) != BMI160_OK)
    {
        std::cout << "init false" << std::endl;
        return;
    }

    // rotation quaternions
    quat quat_integrated{}; // q(t), q_0 = (1,0,0,0)
    quat quat_gyro{};       // instantaneous rotation measured by gyro
    quat quat_correction{}; // correction rotation by accelerometer ("up direction")

    quat_integrated[3] = 1; // Initial Quaternion: (1|0,0,0)
    quat_correction[3] = 1; // Initial Quaternion: (1|0,0,0)

    // raw AccelGyro-Data
    int16_t accelGyro[9] = {0};
    int16_t accelGyroLast[9] = {0};
    double nseconds;

    //Quaternion Calculation Variables
    float theta;
    float inv_omega_len;

    // general positioning calculation
    vec3 gyro_rad_per_s{};
    vec4 accel_m_per_sq_s{};
    accel_m_per_sq_s[3] = 1.0f;

    // debug

    // double cal_value_rotation[3]{};
    // double cal_value_translation[3]{};
    // std::ofstream myfile;
    // myfile.open ("data.txt");

    // vec3 showValue{};

    // std::cout << "Calculating values..." << std::endl;
    //  for (int n = 0;n<OFFSET_MEAS_TIME;n++) {
    //      rslt = bmi160->getAccelGyroData(accelGyro);
    //      std::cout << n << "\t\t";
    //      myfile << n << ";";
    //      for (i = 0; i < 3; i++)
    //      {
    //          cal_value_rotation[i] += (double)accelGyro[i];
    //          // std::cout << (double)accelGyro[i] << ";";
    //          myfile << (double)accelGyro[i] << ";";
    //      }
    //      for (i = 3; i < 6; i++)
    //      {
    //          cal_value_translation[i-3] += (double)accelGyro[i];
    //          showValue[i-3] *= 0.9;
    //          showValue[i-3] += 0.1*accelGyro[i];
    //          std::cout << showValue[i-3] << "    \t";
    //          myfile << (double)accelGyro[i] << ";";
    //      }
    //      std::cout << std::endl;
    //      myfile << std::endl;
    //  }

    //  myfile.close();
    //  std::cout << "Cal values" << std::endl;
    //  for (i = 0; i < 3; i++)
    //  {
    //      // kill offset for gyro
    //      cal_value_rotation[i] /= OFFSET_MEAS_TIME;
    //      cal_value_translation[i] /= OFFSET_MEAS_TIME;
    //      printf("%d: Rotation: %f, Translation: %f\n",i, cal_value_rotation[i],cal_value_translation[i]);

    //  }
    // return;

    zero_rotation_abc[0] = CAL_GYRO_X;
    zero_rotation_abc[1] = CAL_GYRO_Y;
    zero_rotation_abc[2] = CAL_GYRO_Z;
    zero_translation_abc[0] = CAL_ACC_X;
    zero_translation_abc[1] = CAL_ACC_Y;
    zero_translation_abc[2] = CAL_ACC_Z;

    float vec_len{};

    bmi160->getAccelGyroData(accelGyro);
    for (int i = 0; i < 9; i++)
    {
        accelGyroLast[i] = accelGyro[i];
    }

    vec_len = 0.0f;
    for (int i = 0; i < 3; i++)
    {
        gyro_rad_per_s[i] = ((accelGyro[i] - zero_rotation_abc[i]) * GYRO_RESOLUTION) / 180.0 * pi;
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
    for (int i = 0; i < 3; i++)
    {
        vec_len += accel_m_per_sq_s[i] * accel_m_per_sq_s[i];
    }
    vec_len = sqrt(vec_len);

    for (int i = 0; i < 3; i++)
    {
        initial_orientation[i] = accel_m_per_sq_s[i] / vec_len;
    }

    int file = bmi160->I2CGetDataOpenDevice();
    while (1)
    {

        //get both accel and gyro data from bmi160
        //parameter accelGyro is the pointer to store the data
        bmi160->getAccelGyroDataFast(file, accelGyro);
        // get sensor timestamp
        if (accelGyroLast[6] > accelGyro[6])
            nseconds = (double)(accelGyro[6] - accelGyroLast[6] + 65536) * SENS_TIME_RESOLUTION;
        else
            nseconds = (double)(accelGyro[6] - accelGyroLast[6]) * SENS_TIME_RESOLUTION;
        if (GyroDataEqual(accelGyro, accelGyroLast))
            continue;
        // std::cout << nseconds << std::endl;
        // extract all raw measurements with offset-correction
        vec_len = 0.0f;
        //std::cout << "gyro raw values: ( ";
        for (int i = 0; i < 3; i++)
        {
            //std::cout << accelGyro[i] << ", ";
            gyro_rad_per_s[i] = ((accelGyro[i] - zero_rotation_abc[i]) * GYRO_RESOLUTION) / 180.0 * pi;
            accel_m_per_sq_s[i] = ((accelGyro[i + 3] - zero_translation_abc[i]) / ACCEL_RESOLUTION);
        }
        // std::cout << ")" << std::endl;

        // gain correction
        if (accel_m_per_sq_s[0] <= 0)
            accel_m_per_sq_s[0] *= FLIP_BACK_CORR; // x_neg
        else
            accel_m_per_sq_s[0] *= FLIP_FRONT_CORR; // x_pos

        if (accel_m_per_sq_s[1] <= 0)
            accel_m_per_sq_s[1] *= FLIP_RIGHT_CORR; // y_neg
        else
            accel_m_per_sq_s[1] *= FLIP_LEFT_CORR; // y_pos

        if (accel_m_per_sq_s[2] <= 0)
            accel_m_per_sq_s[2] *= FACE_DOWN_CORR; // z_neg
        else
            accel_m_per_sq_s[2] *= FACE_UP_CORR; // z_pos
        // Calculate vector length
        for (int i = 0; i < 3; i++)
        {
            vec_len += accel_m_per_sq_s[i] * accel_m_per_sq_s[i];
        }
        vec_len = sqrt(vec_len);

        // get rotation quaternion from gyro data
        inv_omega_len = Q_rsqrt((float)(gyro_rad_per_s[0] * gyro_rad_per_s[0] + gyro_rad_per_s[1] * gyro_rad_per_s[1] + gyro_rad_per_s[2] * gyro_rad_per_s[2]));
        if (inv_omega_len != 0)
        {
            theta = nseconds / inv_omega_len;
            quat_from_angle_axis(quat_gyro, -theta, gyro_rad_per_s);
        }
        else // no rotation, identity quaternion "does nothing"
        {
            quat_identity(quat_gyro);
        }
        // print_quat("quat_gyro",quat_gyro);
        // update rotation quaternion
        {
            quat temp{};
            quat_mul(temp, quat_gyro, quat_integrated);
            quat_integrated[0] = temp[0];
            quat_integrated[1] = temp[1];
            quat_integrated[2] = temp[2];
            quat_integrated[3] = temp[3];
        }
        // print_quat("quat_integrated",quat_integrated);
        // compute current measured orientation - might be unstable, only apply when stable
        for (int i = 0; i < 3; i++)
        {
            measured_orientation[i] = accel_m_per_sq_s[i] / vec_len;
        }
        // compute orientation as seen by quat_integrated
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
        {
            quat delta_vector_abc{};
            quat quat_integrated_conj{};
            quat temp{};
            for (int i = 0; i < 3; i++)
            {
                delta_vector_abc[i] = accel_m_per_sq_s[i] - updated_orientation[i] * CONST_G;
            }
            quat_conj(quat_integrated_conj, quat_integrated);
            quat_mul(temp, quat_integrated_conj, delta_vector_abc);
            quat_mul(delta_vector_xyz, temp, quat_integrated);
        }

        // if no movement and no rotation: Use accelerometer to find G

        if ((vec_len >= 9.77) && (vec_len <= 9.86) && vec3_check_around_zero(gyro_rad_per_s, NO_ROTATION_TRESH))
        {
            // std::cout << "Stable: \U00002705 (no motion)" << std::endl;
            for (int i = 0; i<3;i++){
                delta_vector_xyz_moving[i] = (1-ACC_FILTER_DEPTH)*delta_vector_xyz_moving[i]+ACC_FILTER_DEPTH*delta_vector_xyz[i];
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
            for (int i = 0; i<3;i++){
               // delta_vector_xyz_moving[i] = (1-ACC_FILTER_DEPTH)*delta_vector_xyz_moving[i]+0.0;
            }            
        }
        for (int i = 0; i<3;i++){
            delta_vector_xyz[i] = delta_vector_xyz[i] - delta_vector_xyz_moving[i];
        }        
        // calculate velocity from acceleration
        for (int i = 0; i < 3; i++)
        {
            velocity_xyz[i] += delta_vector_xyz[i] * nseconds;
        }
        if ((vec_len >= 9.77) && (vec_len <= 9.86) && vec3_check_around_zero(gyro_rad_per_s, NO_ROTATION_TRESH))
        {
            quat_identity(velocity_xyz);
        }
        // calculate position from velocity
        for (int i = 0; i < 3; i++)
        {
            position_xyz[i] += velocity_xyz[i] * nseconds;
        }
        std::cout << std::fixed << std::setprecision(5);
        // std::cout << delta_vector_xyz_moving[0] << ";" << delta_vector_xyz_moving[1] << ";" << delta_vector_xyz_moving[2] << ";";
        // std::cout << delta_vector_xyz[0] << ";" << delta_vector_xyz[1] << ";" << delta_vector_xyz[2] << ";";
        // std::cout << velocity_xyz[0] << ";" << velocity_xyz[1] << ";" << velocity_xyz[2] << ";";
        // std::cout << position_xyz[0] << ";" << position_xyz[1] << ";" << position_xyz[2] << std::endl;
        print_quat("Delta_vector_moving", delta_vector_xyz_moving);
        print_quat("Delta_vector", delta_vector_xyz);
        print_quat("Velocity",velocity_xyz);
        print_quat("Position",position_xyz);
        {
            // update the matrix
            mtx.lock();
            mat4x4_from_quat(quat_matrix, quat_integrated);
            mtx.unlock();
        }
        for (int i = 0; i < 9; i++)
        {
            accelGyroLast[i] = accelGyro[i];
        }
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
