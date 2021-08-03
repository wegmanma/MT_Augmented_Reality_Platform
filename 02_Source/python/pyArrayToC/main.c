#include <stdio.h>
#include <math.h>
int main()
{

    __uint16_t image_x[205][265];
    __uint16_t image_y[205][265];

    char buff[256];
    FILE *latfile;

    sprintf(buff, "%s", "x_corr.dat");
    latfile = fopen(buff, "r");
    fread(&(image_x[0][0]), sizeof(__uint16_t), 205 * 265, latfile);
    fclose(latfile);

    sprintf(buff, "%s", "y_corr.dat");
    latfile = fopen(buff, "r");
    fread(&(image_y[0][0]), sizeof(__uint16_t), 205 * 265, latfile);
    fclose(latfile);
    printf("\n %d     %d     %d     %d     %d     %d     %d", image_x[0][0], image_x[0][1], image_x[0][2], image_x[0][3], image_x[0][4], image_x[0][5], image_x[0][6]);
    printf("\n %d     %d     %d     %d     %d     %d     %d", image_x[1][0], image_x[1][1], image_x[1][2], image_x[1][3], image_x[1][4], image_x[1][5], image_x[1][6]);
    printf("\n %d     %d     %d     %d     %d     %d     %d", image_x[2][0], image_x[2][1], image_x[2][2], image_x[2][3], image_x[2][4], image_x[2][5], image_x[2][6]);
    printf("\n %d     %d     %d     %d     %d     %d     %d", image_x[3][0], image_x[3][1], image_x[3][2], image_x[3][3], image_x[3][4], image_x[3][5], image_x[3][6]);
    printf("\n %d     %d     %d     %d     %d     %d     %d\n", image_x[4][0], image_x[4][1], image_x[4][2], image_x[4][3], image_x[4][4], image_x[4][5], image_x[4][6]);

    printf("\n %d     %d     %d     %d     %d     %d     %d", image_y[0][0], image_y[0][1], image_y[0][2], image_y[0][3], image_y[0][4], image_y[0][5], image_y[0][6]);
    printf("\n %d     %d     %d     %d     %d     %d     %d", image_y[1][0], image_y[1][1], image_y[1][2], image_y[1][3], image_y[1][4], image_y[1][5], image_y[1][6]);
    printf("\n %d     %d     %d     %d     %d     %d     %d", image_y[2][0], image_y[2][1], image_y[2][2], image_y[2][3], image_y[2][4], image_y[2][5], image_y[2][6]);
    printf("\n %d     %d     %d     %d     %d     %d     %d", image_y[3][0], image_y[3][1], image_y[3][2], image_y[3][3], image_y[3][4], image_y[3][5], image_y[3][6]);
    printf("\n %d     %d     %d     %d     %d     %d     %d\n", image_y[4][0], image_y[4][1], image_y[4][2], image_y[4][3], image_y[4][4], image_y[4][5], image_y[4][6]);
}