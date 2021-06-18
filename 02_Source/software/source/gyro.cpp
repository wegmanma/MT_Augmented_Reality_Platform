#include <cstdint>
#include <fcntl.h>
#include <iostream>
#include <linux/i2c-dev.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "../include/BMI160.hpp"

int main(int argc, char **argv)
{
  const int8_t i2c_addr = 0x69;
  BMI160 bmi160;
  if (bmi160.softReset() != BMI160_OK)
  {
      std::cout << "reset false" << std::endl;
  }
  if (bmi160.I2cInit(2, i2c_addr) != BMI160_OK)
  {
    std::cout << "init false" << std::endl;
    return 1;
  }
  
  while (1)
  {
    int i = 0;
    int rslt;
    int16_t accelGyro[6] = {0};

    //get both accel and gyro data from bmi160
    //parameter accelGyro is the pointer to store the data
    rslt = bmi160.getAccelGyroData(accelGyro);
    if (rslt == 0)
    {
      for (i = 0; i < 6; i++)
      {
        if (i < 3)
        {
          //the first three are gyro datas
          std::cout << accelGyro[i] * 3.14 / 180.0 << " ";
        }
        else
        {
          //the following three data are accel datas
          std::cout << accelGyro[i] / 16384.0 << " ";
        }
      }
      std::cout << std::endl;
    }
    else
    {
      std::cout << std::endl;
    }
    // usleep(100000);

    ////only read accel data from bmi160
    //int16_t onlyAccel[3] = {0};
    //bmi160.getAccelData(onlyAccel);
    //{
    //  for (i = 0; i < 3; i++)
    //  {
    //    //the following three data are accel datas
    //    std::cout << onlyAccel[i] / 16384.0 << " ";
    //  }
    //  std::cout << std::endl;
    //}
    //std::cout << std::endl;
  }
  return 0;
}