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
#define ACC_FILTER_DEPTH 0.5 // 1 = no filter, 0 = no update

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

void kalmanCorrectionStep(mat4x17 H_k, mat17x17 P_apriori, mat17x17 P_aposteriori, vec17 x, vec17 x_apriori, quat sourceQuaternion, mat4x4 R_k)
{
    { // ToF Translation Correction: Compute Kalman Gain
        mat17x4 K_Imu;
        // print_mat4x17("ToF H_k", H_k);
        mat17x4 H_k_trans;
        mat4x17_transpose(H_k_trans, H_k);
        mat17x4 PHt;
        mat17x17_mul_mat17x4(PHt, P_apriori, H_k_trans);
        mat4x4 HPHt;
        mat4x17_mul_mat17x4(HPHt, H_k, PHt);
        mat4x4 bracket;
        mat4x4 bracket_inv;
        mat4x4_add(bracket, HPHt, R_k);
        mat4x4_invert(bracket_inv, bracket);
        // print_mat4x4("bracket_inv",bracket_inv);
        mat17x4 K_k;
        mat17x4_mul_mat4x4(K_k, PHt, bracket_inv);
        // ToF Translation Correction: Compute x
        vec4 hx;
        mat4x17_mul_vec17(hx, H_k, x_apriori);
        vec4 bracket2;
        vec4 z_k; // measurement
        {
            z_k[0] = sourceQuaternion[3];
            z_k[1] = sourceQuaternion[0];
            z_k[2] = sourceQuaternion[1];
            z_k[3] = sourceQuaternion[2];
        }
        bracket2[0] = z_k[0] - hx[0];
        bracket2[1] = z_k[1] - hx[1];
        bracket2[2] = z_k[2] - hx[2];
        bracket2[3] = z_k[3] - hx[3];
        vec17 corr_vector;
        mat17x4_mul_vec4(corr_vector, K_k, bracket2);
        vec17_add(x, x_apriori, corr_vector);
        // ToF Translation Correction:  Recalculate Error-Matrix P_k from Kalman-Gain, H and P_apriori
        mat17x17 I;
        mat17x17_identity(I);
        mat17x17 KH;
        mat17x4_mul_mat4x17(KH, K_k, H_k);
        //print_mat17x17("ToF KH",KH);
        mat17x17 P_corr_factor;
        mat17x17_subtract(P_corr_factor, I, KH);
        mat17x17_mul(P_aposteriori, P_corr_factor, P_apriori);
        //print_mat17x17("ToF P_aposteriori",P_aposteriori);
    }
}

void PositionEstimate::get_gyro_matrix(mat4x4 gyro_matrix)
{
    mat4x4 ToFrotation;
    quat ToFQuaterion;
    int newdata;
    float vec_len_i;
    int ToFmtx = tcpCapture->lockMutex();
    newdata = tcpCapture->getRotationTranslation(ToFmtx, ToFrotation, tof_translation_abc);
    tcpCapture->unlockMutex(ToFmtx);
    if (newdata > 0)
    {
        int meas_cnt_i;
        mtx.lock();
        vec4 accel_m_per_sq_s_i;
        meas_cnt_i = meas_cnt;
        accel_m_per_sq_s_i[1] = accel_m_per_sq_s[0] / ((float)meas_cnt);
        accel_m_per_sq_s_i[2] = accel_m_per_sq_s[1] / ((float)meas_cnt);
        accel_m_per_sq_s_i[0] = -accel_m_per_sq_s[2] / ((float)meas_cnt);
        accel_m_per_sq_s_i[3] = (meas_cnt);
        // print_vec4("accel_m_per_sq_s_i", accel_m_per_sq_s_i);
        // std::cout << "get_gyro_matrix: " << nseconds << std::endl;
        double nseconds_i = nseconds;
        vec3 gyro_rad_per_s_i;
        double theta;
        gyro_rad_per_s_i[1] = -gyro_rad_per_s[0] / ((float)meas_cnt);
        gyro_rad_per_s_i[2] = gyro_rad_per_s[1] / ((float)meas_cnt);
        gyro_rad_per_s_i[0] = gyro_rad_per_s[2] / ((float)meas_cnt);
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
        if (meas_cnt_i > 0)
        {
            // Calculate vector length
            vec_len_i = 0.0f;
            for (int i = 0; i < 3; i++)
            {
                vec_len_i += accel_m_per_sq_s_i[i] * accel_m_per_sq_s_i[i];
            }
            vec_len_i = sqrt(vec_len_i);
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
            // print_quat("quat_gyro",quat_gyro);
            quat_from_mat4x4(ToFQuaterion, ToFrotation);

            // compute current measured orientation - might be unstable, only apply when stable


            //print_quat("delta_vector before",delta_vector_xyz);
            //  print_quat("delta_vector_xyz_moving", delta_vector_xyz_moving);

            { // Prediction: Fill F
                F[1][0] = nseconds_i;
                F[2][0] = (nseconds_i * nseconds_i) / 2;
                F[2][1] = nseconds_i;
                F[4][3] = nseconds_i;
                F[5][3] = (nseconds_i * nseconds_i) / 2;
                F[5][4] = nseconds_i;
                F[7][6] = nseconds_i;
                F[8][6] = (nseconds_i * nseconds_i) / 2;
                F[8][7] = nseconds_i;
                F[9][9] = x[13];
                F[9][10] = x[14];
                F[9][11] = x[15];
                F[9][12] = x[16];
                F[10][9] = -x[14];
                F[10][10] = x[13];
                F[10][11] = -x[16];
                F[10][12] = x[15];
                F[11][9] = -x[15];
                F[11][10] = x[16];
                F[11][11] = x[13];
                F[11][12] = -x[14];
                F[12][9] = -x[16];
                F[12][10] = -x[15];
                F[12][11] = x[14];
                F[12][12] = x[13];
            }
            {                                                                    // Prediction: x_t_pred = F * x_t(_past);
                x_apriori[0] = F[0][0] * x[0] + F[1][0] * x[1] + F[2][0] * x[2]; // x
                x_apriori[1] = F[1][1] * x[1] + F[2][1] * x[2];                  // x_dot
                x_apriori[2] = F[2][2] * x[2];
                x_apriori[3] = F[3][3] * x[3] + F[4][3] * x[4] + F[5][3] * x[5]; // x
                x_apriori[4] = F[4][4] * x[4] + F[5][4] * x[5];                  // x_dot
                x_apriori[5] = F[5][5] * x[5];
                x_apriori[6] = F[6][6] * x[6] + F[7][6] * x[7] + F[8][6] * x[8]; // x
                x_apriori[7] = F[7][7] * x[7] + F[8][7] * x[8];                  // x_dot
                x_apriori[8] = F[8][8] * x[8];
                x_apriori[9] = F[9][9] * x[9] + F[10][9] * x[10] + F[11][9] * x[11] + F[12][9] * x[12];
                x_apriori[10] = F[9][10] * x[9] + F[10][10] * x[10] + F[11][10] * x[11] + F[12][10] * x[12];
                x_apriori[11] = F[9][11] * x[9] + F[10][11] * x[10] + F[11][11] * x[11] + F[12][11] * x[12];
                x_apriori[12] = F[9][12] * x[9] + F[10][12] * x[10] + F[11][12] * x[11] + F[12][12] * x[12];
                x_apriori[13] = F[13][13] * x[13];
                x_apriori[14] = F[14][14] * x[14];
                x_apriori[15] = F[15][15] * x[15];
                x_apriori[16] = F[16][16] * x[16];
            }
            { // Prediction Covariance Matrix Apriori
                mat17x17 temp, temp2;
                mat17x17 F_trans;
                mat17x17_mul(temp, F, P_aposteriori);
                mat17x17_transpose(F_trans, F);
                mat17x17_mul(temp2, temp, F_trans);
                mat17x17 Q;
                mat17x17_identity(temp);
                mat17x17_scale(Q, temp, nseconds_i / 2);
                Q[0][0] = pow(nseconds_i,4)/4.0;
                Q[0][1] = pow(nseconds_i,3)/2.0;
                Q[0][2] = pow(nseconds_i,2)/2.0;
                Q[1][0] = pow(nseconds_i,3)/2.0;
                Q[1][1] = pow(nseconds_i,2);
                Q[1][2] = nseconds_i;        
                Q[2][0] = pow(nseconds_i,2)/2.0;
                Q[2][1] = nseconds_i;
                Q[2][2] = 1;

                Q[3][3] = pow(nseconds_i,4)/4.0;
                Q[3][4] = pow(nseconds_i,3)/2.0;
                Q[3][5] = pow(nseconds_i,2)/2.0;
                Q[4][3] = pow(nseconds_i,3)/2.0;
                Q[4][4] = pow(nseconds_i,2);
                Q[4][5] = nseconds_i;        
                Q[5][3] = pow(nseconds_i,2)/2.0;
                Q[5][4] = nseconds_i;
                Q[5][5] = 1;

                Q[6][6] = pow(nseconds_i,4)/4.0;
                Q[6][7] = pow(nseconds_i,3)/2.0;
                Q[6][8] = pow(nseconds_i,2)/2.0;
                Q[7][6] = pow(nseconds_i,3)/2.0;
                Q[7][7] = pow(nseconds_i,2);
                Q[7][8] = nseconds_i;        
                Q[8][6] = pow(nseconds_i,2)/2.0;
                Q[8][7] = nseconds_i;
                Q[8][8] = 1;

                Q[13][13] = Q[13][13] * 50;
                Q[14][14] = Q[14][14] * 50;
                Q[15][15] = Q[15][15] * 50;
                Q[16][16] = Q[16][16] * 50;
                mat17x17_add(P_apriori, temp2, Q);
            }
            {
                mat4x17 H_k = {0};
                H_k[13][0] = 1;
                H_k[14][1] = 1;
                H_k[15][2] = 1;
                H_k[16][3] = 1;
                mat4x4 R = {0};
                mat4x4_identity(R);
                mat4x4_scale(R, R, 0.0001);
                kalmanCorrectionStep(H_k, P_apriori, P_aposteriori, x, x_apriori, quat_gyro, R);
                // print_quat("quat_gyro", quat_gyro,true, true);
            }
            mat17x17_dup(P_apriori, P_aposteriori);
            vec17_dup(x_apriori, x);
            {
                mat4x17 H_k = {0};
                H_k[13][0] = 1;
                H_k[14][1] = 1;
                H_k[15][2] = 1;
                H_k[16][3] = 1;
                // if (newdata == 2) {
                ToFQuaterion[0] = -ToFQuaterion[0];
                ToFQuaterion[1] = -ToFQuaterion[1];
                ToFQuaterion[2] = ToFQuaterion[2];
                ToFQuaterion[3] = ToFQuaterion[3];
                mat4x4 R = {0};
                mat4x4_identity(R);
                mat4x4_scale(R, R, 0.1);
                kalmanCorrectionStep(H_k, P_apriori, P_aposteriori, x, x_apriori, ToFQuaterion, R);
                // print_quat("ToFQuaterion", ToFQuaterion,true);
            }
            // compute orientation as seen by quat_integrated
            {
                quat_integrated[0] = x[10];
                quat_integrated[1] = x[11];
                quat_integrated[2] = x[12];
                quat_integrated[3] = x[9];

                quat quat_integrated_conj{};
                quat temp{};
                quat_conj(quat_integrated_conj, quat_integrated);
                quat_mul(temp, quat_integrated, initial_orientation);
                quat_mul(updated_orientation, temp, quat_integrated_conj);
            }

            // if standing still, compute orientation as seen by accelerometer.
            if ((vec_len_i < 9.81 + 0.2) && (vec_len_i > 9.81 - 0.2))
            {

                for (int i = 0; i < 3; i++)
                {
                    measured_orientation[i] = accel_m_per_sq_s_i[i] / vec_len_i;
                }
                quat_RotationBetweenVectors(quat_correction, updated_orientation, measured_orientation);
                mat17x17_dup(P_apriori, P_aposteriori);
                vec17_dup(x_apriori, x);
                {
                    mat4x17 H_k = {0};
                    H_k[13][0] = 1;
                    H_k[14][1] = 1;
                    H_k[15][2] = 1;
                    H_k[16][3] = 1;
                    mat4x4 R = {0};
                    mat4x4_identity(R);
                    mat4x4_scale(R, R, 0.001);
                    kalmanCorrectionStep(H_k, P_apriori, P_aposteriori, x, x_apriori, quat_correction, R);
                }
                // as the orientation is now changed, update the current orientation
                {
                    quat_integrated[0] = x[10];
                    quat_integrated[1] = x[11];
                    quat_integrated[2] = x[12];
                    quat_integrated[3] = x[9];

                    quat quat_integrated_conj{};
                    quat temp{};
                    quat_conj(quat_integrated_conj, quat_integrated);
                    quat_mul(temp, quat_integrated, initial_orientation);
                    quat_mul(updated_orientation, temp, quat_integrated_conj);
                }
            }

            // rotation complete, now rotate all the translational vectors from abc-coords to xyz-coords

            {

                quat_integrated[0] = x[10];
                quat_integrated[1] = x[11];
                quat_integrated[2] = x[12];
                quat_integrated[3] = x[9];

                quat quat_integrated_conj{};
                quat temp{};
                for (int i = 0; i < 3; i++)
                {
                    delta_vector_abc[i] = accel_m_per_sq_s_i[i] - updated_orientation[i] * CONST_G;
                }
                quat_conj(quat_integrated_conj, quat_integrated);
                quat_mul(temp, quat_integrated_conj, delta_vector_abc);
                quat_mul(delta_vector_xyz, temp, quat_integrated);
                quat_mul(temp, quat_integrated_conj, tof_translation_abc);
                quat_mul(tof_translation_xyz, temp, quat_integrated);
            }
            if (cycle_count < 5)
            {
                delta_vector_xyz[0] = 0.0f;
                delta_vector_xyz[1] = 0.0f;
                delta_vector_xyz[2] = 0.0f;
                delta_vector_xyz[3] = 0.0f;
                tof_translation_xyz[0] = 0.0f;
                tof_translation_xyz[1] = 0.0f;
                tof_translation_xyz[2] = 0.0f;
                tof_translation_xyz[3] = 0.0f;
            }

            // offset correction in a moving average when standing still
            if ((vec_len_i < 9.81 + 0.2) && (vec_len_i > 9.81 - 0.2) && (newdata == 2))
            {
                for (int i = 0; i < 3; i++)
                {
                    delta_vector_xyz_moving[i] = (1 - ACC_FILTER_DEPTH) * delta_vector_xyz_moving[i] + ACC_FILTER_DEPTH * delta_vector_xyz[i];
                    delta_tof_vector_moving[i] = (1 - ACC_FILTER_DEPTH) * delta_tof_vector_moving[i] + ACC_FILTER_DEPTH * tof_translation_xyz[i];
                }
            }
            for (int i = 0; i < 3; i++)
            {
                // delta_vector_xyz[i] = delta_vector_xyz[i] - delta_vector_xyz_moving[i];
                // tof_translation_xyz[i] = tof_translation_xyz[i] - delta_tof_vector_moving[i];
            }// 

            quat delta_vector;
            delta_vector[3] = delta_vector_xyz[0];
            delta_vector[0] = delta_vector_xyz[1];
            delta_vector[1] = delta_vector_xyz[2];
            delta_vector[2] = 0;

            mat17x17_dup(P_apriori, P_aposteriori);
            vec17_dup(x_apriori, x);
            {
                mat4x17 H_k = {0};
                H_k[2][0] = 1;
                H_k[5][1] = 1;
                H_k[8][2] = 1;
                mat4x4 R = {0};
                mat4x4_identity(R);
                mat4x4_scale(R, R, 0.000004);
                // print_vec4("", delta_vector_xyz, true, true);
                kalmanCorrectionStep(H_k, P_apriori, P_aposteriori, x, x_apriori, delta_vector, R);
            }
            quat tof_translation;
            tof_translation_xyz[0] =  tof_translation_xyz[0];
            tof_translation_xyz[1] = -tof_translation_xyz[1];
            tof_translation_xyz[2] =  tof_translation_xyz[2];


            tof_translation[3] = 0.5 * tof_translation_xyz[0];
            tof_translation[0] = 0.5 * tof_translation_xyz[1];
            tof_translation[1] = 0.5 * tof_translation_xyz[2];
            tof_translation[2] = 0;

            mat17x17_dup(P_apriori, P_aposteriori);
            vec17_dup(x_apriori, x);
            {
                mat4x17 H_k = {0};
                H_k[1][0] = 1;
                H_k[4][1] = 1;
                H_k[7][2] = 1;
                mat4x4 R = {0};
                mat4x4_identity(R);
                mat4x4_scale(R, R, 0.15);
                // print_vec4("", tof_translation_xyz, true, true);
                kalmanCorrectionStep(H_k, P_apriori, P_aposteriori, x, x_apriori, tof_translation, R);
            }


            // print_vec17("X3", x, true);
            //    print_quat("quat_integrated", quat_integrated);
            //    // std::cout << "------------------------------------------------------------------" << cycle_count << "--------------------------------------------------" << std::endl;
            cycle_count++;
            if (cycle_count > 3000)
                cycle_count = 5;
            // std::cout << vec_len_i << ";" << meas_cnt_i << ";";
            // print_quat("quat_integrated", quat_integrated, true, true);
            // 
            // print_quat("ToFquaternion", ToFQuaterion, true, true);
            //    //print_mat4x4("quat_matrix",quat_matrix);

            kalman_rotation[0] = x[10];
            kalman_rotation[1] = x[11];
            kalman_rotation[2] = x[12];
            kalman_rotation[3] = x[9];

            kalman_translation[0] = x[0];
            kalman_translation[1] = x[1];
            kalman_translation[2] = x[2];
            kalman_translation[3] = 0;

            quat kalman_velocity;
            kalman_velocity[0] = x[3];
            kalman_velocity[1] = x[4];
            kalman_velocity[2] = x[5];
            kalman_velocity[3] = 0;

            kalman_translation_from_velocity[0] += kalman_velocity[0]*nseconds_i;
            kalman_translation_from_velocity[1] += kalman_velocity[1]*nseconds_i;
            kalman_translation_from_velocity[2] += kalman_velocity[2]*nseconds_i;
            kalman_translation_from_velocity[3] = 0;

            quat kalman_acceleration;
            kalman_acceleration[0] = x[6];
            kalman_acceleration[1] = x[7];
            kalman_acceleration[2] = x[8];
            kalman_acceleration[3] = 0;
            // print_vec4("kalman_acceleration", kalman_acceleration, true, true);
            // print_vec4("kalman_velocity", kalman_velocity, true, true);
            // print_vec4("kalman_translation", kalman_translation, true, true);
            // ;
            // 

            quat kalman_rot_speed = {x[14], x[15], x[16], x[13]};
            std::cout << nseconds_i << ";";
            print_quat("quat_gyro", quat_gyro, true, true);
            print_quat("ToFquaternion", ToFQuaterion, true, true);
            print_quat("kalman_rotation", kalman_rotation, true, true);
            print_quat("kalman_rot_speed", kalman_rot_speed, true, true);
            print_quat("delta_vector_xyz", delta_vector_xyz, true, true);
            print_vec4("tof_translation_xyz", tof_translation_xyz, true, true);
            print_quat("kalman_translation", kalman_translation, true, true);
            print_quat("kalman_velocity", kalman_velocity, true, true);
            print_quat("kalman_acceleration", kalman_acceleration, true, true);
            print_vec4("kalman_translation_from_velocity", kalman_translation_from_velocity, true);
            // 
        }
    }
    if (ToFQuaterion[3] > 0.75)
    {
        quat temp;
        quat_mul(temp, quat_tof_integrated, ToFQuaterion);
        quat_tof_integrated[0] = temp[0];
        quat_tof_integrated[1] = temp[1];
        quat_tof_integrated[2] = temp[2];
        quat_tof_integrated[3] = temp[3];
    }

    // update the matrix
    mat4x4_from_quat(quat_matrix, kalman_rotation);
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
    x_apriori[10] = 1.0;
    x_apriori[13] = 1.0;
    x[9] = 1.0;
    x[13] = 1.0;

    for (int i = 0; i < 17; i++)
    {
        for (int j = 0; j < 17; j++)
        {
            P_apriori[i][j] = i == j ? 1.f : 0.f;
            P_aposteriori[i][j] = i == j ? 1.f : 0.f;
            F[i][j] = i == j ? 1.f : 0.f;
        }
    }

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
        for (int i = 0; i < 3; i++)
        {
            vec_len += accel_i[i] * accel_i[i];
        }
        vec_len = sqrt(vec_len);
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

        nseconds += nseconds_i;
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
