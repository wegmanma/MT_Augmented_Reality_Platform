#include <iostream>
#include <thread>

#include <PositionEstimate.hpp>

PositionEstimate::PositionEstimate()
{
    bmi160 = new BMI160(2, 0x69);
    tid = std::thread(&PositionEstimate::thrBMI160, this);
    tid.detach();
}

PositionEstimate::~PositionEstimate()
{
}

void PositionEstimate::get_gyro_data(vec3 gyro_data) {
    
    gyro_data[0] = cumulated_rotation_xyz[0];
    gyro_data[1] = cumulated_rotation_xyz[1];
    gyro_data[2] = cumulated_rotation_xyz[2];
}

void PositionEstimate::thrBMI160()
{

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
    
    rslt = bmi160->getAccelGyroData(accelGyro);
    if (rslt == 0)
    {
        for (i = 0; i < 3; i++)
        {
                // kill offset for gyro
                zero_rotation_xyz[i] = accelGyro[i] * 3.14 / 180.0;      
        }
    }


    while (1)
    {

        //get both accel and gyro data from bmi160
        //parameter accelGyro is the pointer to store the data
        rslt = bmi160->getAccelGyroData(accelGyro);
        for (i = 0; i < 3; i++)
        {
                // kill offset for gyro
                cumulated_rotation_xyz[i] += (accelGyro[i] * 3.14 / 180.0);      
        }
        //if (rslt == 0)
        //{
        //    for (i = 0; i < 6; i++)
        //    {
        //        if (i < 3)
        //        {
        //            //the first three are gyro datas
        //            std::cout << (accelGyro[i] * 3.14 / 180.0) - zero_rotation_xyz[i]<< " ";
        //        }
        //        else
        //        {
        //            //the following three data are accel datas
        //            std::cout << accelGyro[i] / 16384.0 << " ";
        //        }
        //    }
        //    std::cout << std::endl;
        //}
        //else
        //{
        //    std::cout << std::endl;
        //}
    }
}
