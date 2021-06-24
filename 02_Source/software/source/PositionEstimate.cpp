#include <iostream>
#include <fstream>
#include <thread>

#include <PositionEstimate.hpp>

#define CONST_G 9.81

#define NO_ROTATION_TRESH 0.02
#define NO_MOVEMENT_MAX_COUNT 5

#define GYRO_RESOLUTION 0.061
#define ACCEL_RESOLUTION 1670.13*8

#define SENS_TIME_RESOLUTION 0.000039

#define OFFSET_MEAS_TIME 1000
#define CAL_GYRO_X 2.381310
#define CAL_GYRO_Y -2.783860
#define CAL_GYRO_Z 6.784730
#define CAL_ACC_X -65.8768
#define CAL_ACC_Y -90.7988
#define CAL_ACC_Z 41.24667

bool check_around_zero(float val, float tresh) {
    if ((val < tresh)&&(val > (-1.0)*tresh)) return true;
    else return false;
}

bool vec3_check_around_zero(vec3 val, float tresh) {
    for (int i = 0; i<3; i++) {
        if (!check_around_zero(val[i], tresh)) return false;
    }
    return true;
}

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
    if (bmi160->I2cInit(8, 0x69) != BMI160_OK)
    {
        std::cout << "init false" << std::endl;
        return;
    }

    int i = 0;
    int rslt;
    // raw AccelGyro-Data
    int16_t accelGyro[9] = {0};
    int16_t sensorTime_last = {0};
    double nseconds;

    //Quaternion Calculation Variables
    float theta;
    float inv_omega_len;
    quat temp_quat;
    vec3 norm_orientation;

    // general positioning calculation
    vec4 up_direction_abc = {0.0f, 0.0f, 0.0f, 1.0f};
    vec3 speed_abc;
    vec3 pos_abc;
    vec3 gyro_rad_per_s;
    float LPF_Beta = 0.1; // 0<ß<1
    vec4 accel_m_per_sq_s_raw{};    
    vec4 accel_m_per_sq_s{};
    accel_m_per_sq_s[3] = 1.0f;
    vec4 sensor_position{};
    sensor_position[3] = 1.0f;
    // no movement detection

    int no_movement_counter{};


    // debug

    vec3 max{}, min{};


    mat4x4_identity(cumulated_mat);
    mat4x4_identity(temp_mat);
    double cal_value_rotation[3]{};
    double cal_value_translation[3]{};
    std::ofstream myfile;
    myfile.open ("data.txt");
    
    /* vec3 showValue{};
    
    std::cout << "Calculating values..." << std::endl;
     for (int n = 0;n<OFFSET_MEAS_TIME;n++) {
         rslt = bmi160->getAccelGyroData(accelGyro);
         std::cout << n << "\t\t";
         myfile << n << ";";
         for (i = 0; i < 3; i++)
         {
             cal_value_rotation[i] += (double)accelGyro[i];
             // std::cout << (double)accelGyro[i] << ";";
             myfile << (double)accelGyro[i] << ";";
         }
         for (i = 3; i < 6; i++)
         {
             cal_value_translation[i-3] += (double)accelGyro[i];
             showValue[i-3] *= 0.9;
             showValue[i-3] += 0.1*accelGyro[i];
             std::cout << showValue[i-3] << "    \t";
             myfile << (double)accelGyro[i] << ";";
         }
         std::cout << std::endl;
         myfile << std::endl;
     }

     myfile.close();
     std::cout << "Cal values" << std::endl;
     for (i = 0; i < 3; i++)
     {
         // kill offset for gyro
         cal_value_rotation[i] /= OFFSET_MEAS_TIME;
         cal_value_translation[i] /= OFFSET_MEAS_TIME;
         printf("%d: Rotation: %f, Translation: %f\n",i, cal_value_rotation[i],cal_value_translation[i]);


     }
    return;
     zero_translation_abc[2] -= 16384; */
    zero_rotation_abc[0] = CAL_GYRO_X;
    zero_rotation_abc[1] = CAL_GYRO_Y;
    zero_rotation_abc[2] = CAL_GYRO_Z;
    zero_translation_abc[0] = CAL_ACC_X;
    zero_translation_abc[1] = CAL_ACC_Y;
    zero_translation_abc[2] = CAL_ACC_Z;

    float vec_len{};

    // Calibration step ("up direction initially depends on Accelerometer (Gravity)")
    rslt = bmi160->getAccelGyroData(accelGyro);
    sensorTime_last = accelGyro[7]; 
    vec_len = 0.0f;
    for (i = 0; i < 3; i++)
    {
        gyro_rad_per_s[i] = ((accelGyro[i] - zero_rotation_abc[i]) * GYRO_RESOLUTION) / 180.0 * pi;
        accel_m_per_sq_s[i] = (accelGyro[i + 3] - zero_translation_abc[i]) / ACCEL_RESOLUTION;
        vec_len += accel_m_per_sq_s[i] * accel_m_per_sq_s[i];
    }
    vec_len = sqrt(vec_len);

    for (i = 0; i < 3; i++)
    {
        up_direction_abc[i] = accel_m_per_sq_s[i] / vec_len;
        std::cout << up_direction_abc[i] << ", ";
    }
    
    float height = 0.0f;
    while (1)
    {

        //get both accel and gyro data from bmi160
        //parameter accelGyro is the pointer to store the data
        startTime = std::chrono::steady_clock::now();
        rslt = bmi160->getAccelGyroData(accelGyro);
        endTime = std::chrono::steady_clock::now();
        timeSpan = endTime - startTime;
        nseconds = double(timeSpan.count()) * std::chrono::steady_clock::period::num / std::chrono::steady_clock::period::den;
        
        if (sensorTime_last > accelGyro[6]) { // wrap around
            std::cout << "nseconds: " << nseconds << " Sensortime:" << (double)(accelGyro[6]-sensorTime_last+65536)*SENS_TIME_RESOLUTION << " Sensortime raw:" << accelGyro[6] << " WRAP WRAP WRAP!" << std::endl;
            nseconds = (double)(accelGyro[6]-sensorTime_last+65536)*SENS_TIME_RESOLUTION;
        } else {
            std::cout << "nseconds: " << nseconds << " Sensortime:" << (double)(accelGyro[6]-sensorTime_last)*SENS_TIME_RESOLUTION << " Sensortime raw:" << accelGyro[6] << std::endl;
            nseconds = (double)(accelGyro[6]-sensorTime_last)*SENS_TIME_RESOLUTION;
        }
        // std::cout << "Accel" << std::endl;
        vec_len = 0.0f;
        for (i = 0; i < 3; i++)
        {
            gyro_rad_per_s[i] = ((accelGyro[i] - zero_rotation_abc[i]) * GYRO_RESOLUTION) / 180.0 * pi;
            accel_m_per_sq_s_raw[i] = ((accelGyro[i + 3] - zero_translation_abc[i]) / ACCEL_RESOLUTION);
            accel_m_per_sq_s[i] = accel_m_per_sq_s[i] - (LPF_Beta * (accel_m_per_sq_s[i] - accel_m_per_sq_s_raw[i]));
            vec_len += accel_m_per_sq_s[i] * accel_m_per_sq_s[i];
            std::cout << gyro_rad_per_s[i] << " rad/s" << std::endl;
            std::cout << accel_m_per_sq_s[i] << " m/s^2" << std::endl;
        }
        vec_len = sqrt(vec_len);

        // cross sensitivity correction
        if (accel_m_per_sq_s[0]<=0) { // x_neg
        std::cout << "x neg: " << accel_m_per_sq_s[0]*0.00143 << std::endl;
        vec_len += accel_m_per_sq_s[0]*0.00143;
        } else { // x_pos
        std::cout << "x pos: " << accel_m_per_sq_s[0]*-0.000143 << std::endl;
        vec_len += accel_m_per_sq_s[0]*-0.000143;
        }
        if (accel_m_per_sq_s[1]<=0) { // y_neg
        std::cout << "y neg: " << accel_m_per_sq_s[1]*0.005872 << std::endl;
        vec_len += accel_m_per_sq_s[1]*0.005872;
        } else { // y_pos
        std::cout << "y pos: " << accel_m_per_sq_s[1]*-0.008039 << std::endl;
        vec_len += accel_m_per_sq_s[1]*-0.008039;            
        }
        if (accel_m_per_sq_s[2]<=0) { // z_neg
        std::cout << "z neg: " << accel_m_per_sq_s[2]*0.010122 << std::endl;
        vec_len += accel_m_per_sq_s[2]*0.010122;
        } else { // z_pos
        std::cout << "z pos: " << accel_m_per_sq_s[2]*-0.01125 << std::endl;      
        vec_len += accel_m_per_sq_s[2]*-0.01125;
        }
        


        inv_omega_len = Q_rsqrt((float)(gyro_rad_per_s[0] * gyro_rad_per_s[0] + gyro_rad_per_s[1] * gyro_rad_per_s[1] + gyro_rad_per_s[2] * gyro_rad_per_s[2]));
        if (inv_omega_len != 0)
        {
            theta = nseconds / inv_omega_len;
            norm_orientation[0] = gyro_rad_per_s[0] * inv_omega_len;
            norm_orientation[1] = gyro_rad_per_s[1] * inv_omega_len;
            norm_orientation[2] = gyro_rad_per_s[2] * inv_omega_len;
            quat_gyro[0] = cos(theta / 2);
            quat_gyro[1] = norm_orientation[0] * sin(theta / 2);
            quat_gyro[2] = norm_orientation[1] * sin(theta / 2);
            quat_gyro[3] = norm_orientation[2] * sin(theta / 2);
            //====================================

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
            mat4x4_mul(temp_mat, quat_matrix, cumulated_mat);

            mat4x4_dup(cumulated_mat, temp_mat);
            cumulated_mat[3][3] = 1.0f;
            std::cout << "cumulated_mat = " << std::endl;
            std::cout << "|" << cumulated_mat[0][0] << " " << cumulated_mat[1][0] << " " << cumulated_mat[2][0] << " " << cumulated_mat[3][0] << "|" << std::endl;
            std::cout << "|" << cumulated_mat[0][1] << " " << cumulated_mat[1][1] << " " << cumulated_mat[2][1] << " " << cumulated_mat[3][1] << "|" << std::endl;
            std::cout << "|" << cumulated_mat[0][2] << " " << cumulated_mat[1][2] << " " << cumulated_mat[2][2] << " " << cumulated_mat[3][2] << "|" << std::endl;
            std::cout << "|" << cumulated_mat[0][3] << " " << cumulated_mat[1][3] << " " << cumulated_mat[2][3] << " " << cumulated_mat[3][3] << "|" << std::endl;
        }

        // else use gyro to find rotation change
        else {
            vec4 temp;
            mat4x4_mul_vec4(temp, quat_matrix, up_direction_abc);
            up_direction_abc[0] = temp[0];
            up_direction_abc[1] = temp[1];
            up_direction_abc[2] = temp[2];
        }
        std::cout << "Up: (";
        for (i = 0; i < 3; i++)
        {
            std::cout << up_direction_abc[i] << ", ";
        }
        std::cout << ")" << std::endl;
        std::cout << "vector length = " << vec_len << std::endl;

        vec4 delta_vector_xyz{};  
        vec4 delta_temp_vector_abc{};
        mat4x4 cumulated_inverted{};  
        mat4x4_invert(cumulated_inverted,cumulated_mat);    
        std::cout << "cumulated_inverted = " << std::endl;
        std::cout << "|" << cumulated_inverted[0][0] << " " << cumulated_inverted[1][0] << " " << cumulated_inverted[2][0] << " " << cumulated_inverted[3][0] << "|" << std::endl;
        std::cout << "|" << cumulated_inverted[0][1] << " " << cumulated_inverted[1][1] << " " << cumulated_inverted[2][1] << " " << cumulated_inverted[3][1] << "|" << std::endl;
        std::cout << "|" << cumulated_inverted[0][2] << " " << cumulated_inverted[1][2] << " " << cumulated_inverted[2][2] << " " << cumulated_inverted[3][2] << "|" << std::endl;
        std::cout << "|" << cumulated_inverted[0][3] << " " << cumulated_inverted[1][3] << " " << cumulated_inverted[2][3] << " " << cumulated_inverted[3][3] << "|" << std::endl;
        delta_vector_xyz[3] = 1.0f;    
        for (i = 0; i < 3; i++)
        {
              delta_temp_vector_abc[i] =  accel_m_per_sq_s[i] - up_direction_abc[i]*CONST_G;
        }
        mat4x4_mul_vec4(delta_vector_xyz,cumulated_inverted, delta_temp_vector_abc);
        // if no movement and no rotation: Use accelerometer to find G

        if ((vec_len >= 9.77) && (vec_len <= 9.86) && vec3_check_around_zero(gyro_rad_per_s, NO_ROTATION_TRESH))
        {
            for (i = 0; i < 3; i++)
            {
                up_direction_abc[i] = accel_m_per_sq_s[i] / vec_len;
            }
            no_movement_counter++;
            if (no_movement_counter >= NO_MOVEMENT_MAX_COUNT) {
                std::cout << "zeroing: yes =================================================================" << std::endl;
                speed_abc[0] = 0.0f;
                speed_abc[1] = 0.0f;
                speed_abc[2] = 0.0f;  
                delta_vector_xyz[0] = 0.0f;     
                delta_vector_xyz[1] = 0.0f;       
                delta_vector_xyz[2] = 0.0f;                
            } else
                std::cout << "zeroing: no, counting " << no_movement_counter << std::endl;
        } else {
            no_movement_counter = 0;
            std::cout << "zeroing: no" << std::endl;
        }
        std::cout << "delta: (";
        for (i = 0; i < 3; i++)
        {        
            if (max[i] < delta_vector_xyz[i]) max[i] = delta_vector_xyz[i];
            if (min[i] > delta_vector_xyz[i]) min[i] = delta_vector_xyz[i];
            speed_abc[i] += delta_vector_xyz[i]*nseconds;
            pos_abc[i] += speed_abc[i]*nseconds;
            std::cout << delta_vector_xyz[i] << ", ";
        }
        std::cout << ")" << std::endl;
        std::cout << "speed: (";
        for (i = 0; i < 3; i++)
        {
            std::cout << speed_abc[i] << ", ";
        }
        std::cout << ")" << std::endl;
        std::cout << "pos: (";
        for (i = 0; i < 3; i++)
        {
            std::cout << pos_abc[i] << ", ";
        }
        std::cout << ")" << std::endl;
        std::cout << "max: (";
        for (i = 0; i < 3; i++)
        {
            std::cout << max[i] << ", ";
        }
        std::cout << ")" << std::endl;
        std::cout << "min: (";
        for (i = 0; i < 3; i++)
        {
            std::cout << min[i] << ", ";
        }
        std::cout << ")" << std::endl;
        std::cout << pos_abc[2] << std::endl;
        sensorTime_last = accelGyro[6];
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